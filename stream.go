// Package v4l, a facade to the Video4Linux video capture interface
// Copyright (C) 2016 Zoltán Korándi <korandi.z@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

//go:build linux
// +build linux

package v4l

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// A Frame is a single video frame delivered by a Stream. Unlike Buffer, a
// Frame owns its data: the image bytes are copied out of the device buffer
// before that buffer is handed back to the device, so a Frame remains valid
// for as long as the caller keeps a reference to it, regardless of any
// subsequent captures, TurnOff, or Close calls.
type Frame struct {
	// Data holds the raw image data in the format described by Config.
	Data []byte

	// SeqNum is the frame sequence number reported by the kernel.
	SeqNum uint32

	// Timestamp is the capture time on the device clock. If the device did
	// not provide a usable timestamp, Timestamp is the reception time
	// measured from the start of the stream and TimestampFallback is true.
	Timestamp time.Duration

	// TimestampFallback tells whether Timestamp had to be synthesized
	// locally because the device provided none.
	TimestampFallback bool

	// Config is the device configuration the frame was captured with.
	Config DeviceConfig

	// Info describes how the image data is laid out in Data.
	Info BufferInfo
}

// DropPolicy determines what a Stream does with a freshly captured frame
// when its queue is full.
type DropPolicy int

const (
	// DropOldest discards the oldest queued frame to make room for the new
	// one. The capture thread never blocks on a full queue, so a slow
	// consumer keeps receiving the most recent frames at the cost of gaps.
	DropOldest DropPolicy = iota

	// BlockProducer makes the capture thread wait until the consumer frees
	// up space in the queue. No frames are lost, but a slow consumer slows
	// down the whole pipeline.
	BlockProducer
)

// A StreamConfig configures a Stream.
type StreamConfig struct {
	// QueueCapacity is the maximum number of frames held in the stream's
	// queue. A capacity of zero creates an unbuffered stream: with
	// DropOldest every frame is dropped unless a consumer is already
	// waiting for it, while with BlockProducer each frame is handed over
	// directly to a consumer. Negative values are invalid.
	QueueCapacity int

	// Policy determines the behavior when the queue is full.
	Policy DropPolicy
}

// A Stream continuously captures frames from a Device in the background and
// hands them to a consumer through Next. Because every frame is copied out
// of its device buffer and the buffer is immediately returned to the device,
// the stream never holds device memory: a slow or cancelled consumer cannot
// stall the device-side buffer ring, and no buffers leak when the consumer
// goes away or the device is stopped.
//
// A Stream must be closed with Close when no longer needed. Turning the
// device off, closing it, or unplugging it terminates the stream: Next then
// reports the cause via the returned error (also available through Err).
// The stream must not be used concurrently with Device.Capture.
type Stream struct {
	src    frameSource
	frames chan *Frame
	done   chan struct{} // closed by Close
	fin    chan struct{} // closed when the capture goroutine exits

	closeOnce sync.Once
	block     bool

	mu  sync.Mutex
	err error // terminal capture error, set before fin is closed

	cfg   DeviceConfig
	info  BufferInfo
	start time.Time

	dropped uint64 // atomic
}

// NewStream starts a background capture stream on the device, configured by
// sc. The device must be turned on. The stream snapshots the current device
// configuration and attaches it to every frame it produces; if the device
// configuration changes mid-stream (detected when a frame no longer fits
// the known layout), the snapshot is refreshed and subsequent frames carry
// the new format information.
func (d *Device) NewStream(sc StreamConfig) (*Stream, error) {
	if sc.QueueCapacity < 0 {
		return nil, Error("negative queue capacity")
	}
	if len(d.buffers) == 0 {
		return nil, Error("device is not capturing")
	}
	cfg, err := d.GetConfig()
	if err != nil {
		return nil, err
	}
	info, err := d.BufferInfo()
	if err != nil {
		return nil, err
	}
	return newStream(deviceFrameSource{d}, sc, cfg, info), nil
}

// newStream starts a capture stream reading from src.
func newStream(src frameSource, sc StreamConfig, cfg DeviceConfig, info BufferInfo) *Stream {
	if sc.QueueCapacity < 0 {
		sc.QueueCapacity = 0
	}
	s := &Stream{
		src:    src,
		frames: make(chan *Frame, sc.QueueCapacity),
		done:   make(chan struct{}),
		fin:    make(chan struct{}),
		block:  sc.Policy == BlockProducer,
		cfg:    cfg,
		info:   info,
		start:  time.Now(),
	}
	go s.run()
	return s
}

// Next returns the next frame from the stream, blocking until one is
// available, the stream terminates, or ctx is cancelled. Queued frames are
// delivered even after the stream has terminated; the terminal error is
// reported only once the queue has been drained. After the stream is
// closed, Next fails with ErrStreamClosed. A cancelled ctx cancels only
// this call; it never affects the stream itself or other consumers.
func (s *Stream) Next(ctx context.Context) (*Frame, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	for {
		// Drain queued frames first: they stay valid after termination.
		select {
		case f := <-s.frames:
			return f, nil
		default:
		}
		// Terminal states with an empty queue.
		select {
		case <-s.fin:
			return nil, s.Err()
		default:
		}
		select {
		case <-s.done:
			return nil, ErrStreamClosed
		default:
		}
		// Wait for something to happen.
		select {
		case f := <-s.frames:
			return f, nil
		case <-s.fin:
		case <-s.done:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

// Err returns the error that terminated the stream, if any. A stream
// terminated by Close reports ErrStreamClosed; a device failure (such as
// the device being turned off, closed, or unplugged) is reported as the
// underlying error.
func (s *Stream) Err() error {
	s.mu.Lock()
	err := s.err
	s.mu.Unlock()
	if err != nil {
		return err
	}
	select {
	case <-s.done:
		return ErrStreamClosed
	default:
		return nil
	}
}

// Dropped returns the number of frames discarded so far, either because the
// queue was full (see DropPolicy) or because the device flagged them as
// corrupt.
func (s *Stream) Dropped() uint64 {
	return atomic.LoadUint64(&s.dropped)
}

// Close shuts the stream down. It is idempotent and safe to call
// concurrently with Next and with itself. Close waits for the capture
// goroutine to finish; if the goroutine is blocked waiting for a frame from
// a device that has stopped producing, turning the device off unblocks it.
// Frames still queued at the time of the call remain available through Next
// until drained.
func (s *Stream) Close() {
	s.closeOnce.Do(func() { close(s.done) })
	<-s.fin
}

// run is the capture loop. It dequeues device buffers, copies their
// contents into frames, returns the buffers to the device, and delivers the
// frames to the consumer.
func (s *Stream) run() {
	defer close(s.fin)
	for {
		select {
		case <-s.done:
			return
		default:
		}
		raw, err := s.src.dequeue()
		if err != nil {
			select {
			case <-s.done:
				// Closed concurrently; the error is a side effect.
			default:
				s.setErr(err)
			}
			return
		}
		if raw.flags&v4l_bufFlagError != 0 {
			// Corrupt frame: return the buffer and move on.
			atomic.AddUint64(&s.dropped, 1)
			if err := s.src.requeue(raw.index); err != nil {
				s.setErr(err)
				return
			}
			continue
		}
		f := s.makeFrame(raw)
		// Hand the device buffer back before queueing the frame. The
		// frame owns a copy of the data, so nothing downstream holds
		// device memory.
		if err := s.src.requeue(raw.index); err != nil {
			s.setErr(err)
			return
		}
		if !s.deliver(f) {
			return
		}
	}
}

// makeFrame copies raw into a Frame and attaches timestamp and format
// information.
func (s *Stream) makeFrame(raw rawFrame) *Frame {
	data := make([]byte, len(raw.data))
	copy(data, raw.data)

	f := &Frame{Data: data, SeqNum: raw.seq}

	ts := raw.timestamp
	if (ts.sec == 0 && ts.usec == 0) ||
		raw.flags&v4l_bufFlagTimestampMask == v4l_bufFlagTimestampUnknown {
		// The device did not provide a usable timestamp; fall back to
		// the local clock.
		f.Timestamp = time.Since(s.start)
		f.TimestampFallback = true
	} else {
		f.Timestamp = time.Duration(ts.sec)*time.Second +
			time.Duration(ts.usec)*time.Microsecond
	}

	// A frame that does not fit the format the stream was started with
	// indicates that the device configuration changed underneath us.
	// Refresh the cached format information so that this and subsequent
	// frames are described correctly.
	if s.info.BufferSize > 0 && len(raw.data) > s.info.BufferSize {
		if fq, ok := s.src.(formatQuerier); ok {
			if cfg, info, err := fq.format(); err == nil {
				s.cfg, s.info = cfg, info
			}
		}
	}
	f.Config = s.cfg
	f.Info = s.info
	return f
}

// deliver queues f for the consumer according to the stream's drop policy.
// It returns false if the stream was closed in the meantime.
func (s *Stream) deliver(f *Frame) bool {
	if s.block {
		select {
		case s.frames <- f:
			return true
		case <-s.done:
			return false
		}
	}
	// DropOldest.
	select {
	case s.frames <- f:
		return true
	default:
	}
	select {
	case <-s.frames:
		// Dropped the oldest queued frame to make room.
		atomic.AddUint64(&s.dropped, 1)
		select {
		case s.frames <- f:
		case <-s.done:
			return false
		}
	default:
		// Unbuffered stream with no waiting consumer: drop the new
		// frame itself.
		atomic.AddUint64(&s.dropped, 1)
	}
	return true
}

// setErr records the terminal error of the stream.
func (s *Stream) setErr(err error) {
	s.mu.Lock()
	s.err = err
	s.mu.Unlock()
}

// A rawFrame is a frame as dequeued from a frameSource. Its data is only
// valid until the buffer is returned with requeue.
type rawFrame struct {
	index     uint32
	data      []byte
	seq       uint32
	timestamp v4l_timeval
	flags     uint32
}

// A frameSource abstracts the device-side of a Stream: it dequeues captured
// buffers and returns them to the device.
type frameSource interface {
	dequeue() (rawFrame, error)
	requeue(index uint32) error
}

// A formatQuerier is an optional interface implemented by frame sources
// that can report the current device configuration.
type formatQuerier interface {
	format() (DeviceConfig, BufferInfo, error)
}

// deviceFrameSource is the frameSource backed by a real Device.
type deviceFrameSource struct {
	d *Device
}

func (s deviceFrameSource) dequeue() (rawFrame, error) {
	b := v4l_buffer{
		typ:    v4l_bufTypeVideoCapture,
		memory: v4l_memoryMmap,
	}
	if err := ioctl_dqbuf(s.d.fd, &b); err != nil {
		return rawFrame{}, err
	}
	if int(b.index) >= len(s.d.buffers) {
		return rawFrame{}, Error("dequeued buffer index out of range")
	}
	data := s.d.buffers[b.index]
	if int(b.bytesused) < len(data) {
		data = data[:b.bytesused]
	}
	return rawFrame{
		index:     b.index,
		data:      data,
		seq:       b.sequence,
		timestamp: b.timestamp,
		flags:     b.flags,
	}, nil
}

func (s deviceFrameSource) requeue(index uint32) error {
	b := v4l_buffer{
		typ:    v4l_bufTypeVideoCapture,
		memory: v4l_memoryMmap,
		index:  index,
	}
	return ioctl_qbuf(s.d.fd, &b)
}

func (s deviceFrameSource) format() (DeviceConfig, BufferInfo, error) {
	cfg, err := s.d.GetConfig()
	if err != nil {
		return DeviceConfig{}, BufferInfo{}, err
	}
	info, err := s.d.BufferInfo()
	if err != nil {
		return DeviceConfig{}, BufferInfo{}, err
	}
	return cfg, info, nil
}
