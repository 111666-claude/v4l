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
	"io"
	"sync"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

// fakeFrame describes one frame in a scripted buffer sequence.
type fakeFrame struct {
	index uint32
	data  []byte
	seq   uint32
	ts    v4l_timeval
	flags uint32
}

// fakeSource is a frameSource feeding a Stream a scripted sequence of
// frames from simulated device buffers. Buffers are reused between frames
// and overwritten on requeue, mimicking real mmap'd device buffers.
type fakeSource struct {
	mu       sync.Mutex
	cond     *sync.Cond
	pending  []fakeFrame
	err      error // returned by dequeue once pending is drained
	buffers  [][]byte
	dequeued map[uint32]bool // buffers currently checked out
	requeued []uint32
	formats  []formatReply // scripted format() replies
}

type formatReply struct {
	cfg  DeviceConfig
	info BufferInfo
	err  error
}

func newFakeSource(nbuf, bufsize int) *fakeSource {
	f := &fakeSource{
		buffers:  make([][]byte, nbuf),
		dequeued: make(map[uint32]bool),
	}
	f.cond = sync.NewCond(&f.mu)
	for i := range f.buffers {
		f.buffers[i] = make([]byte, bufsize)
	}
	return f
}

// feed appends frames to the sequence the source will produce.
func (f *fakeSource) feed(frames ...fakeFrame) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pending = append(f.pending, frames...)
	f.cond.Broadcast()
}

// finish makes dequeue return err once the pending frames are drained.
func (f *fakeSource) finish(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.err = err
	f.cond.Broadcast()
}

func (f *fakeSource) dequeue() (rawFrame, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for len(f.pending) == 0 && f.err == nil {
		f.cond.Wait()
	}
	if len(f.pending) == 0 {
		return rawFrame{}, f.err
	}
	fr := f.pending[0]
	f.pending = f.pending[1:]
	buf := f.buffers[fr.index]
	copy(buf, fr.data)
	f.dequeued[fr.index] = true
	return rawFrame{
		index:     fr.index,
		data:      buf[:len(fr.data)],
		seq:       fr.seq,
		timestamp: fr.ts,
		flags:     fr.flags,
	}, nil
}

func (f *fakeSource) requeue(index uint32) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.dequeued, index)
	f.requeued = append(f.requeued, index)
	// Simulate the device scribbling over the returned buffer.
	for i := range f.buffers[index] {
		f.buffers[index][i] = 0xff
	}
	return nil
}

func (f *fakeSource) format() (DeviceConfig, BufferInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.formats) == 0 {
		return DeviceConfig{}, BufferInfo{}, Error("no scripted format")
	}
	r := f.formats[0]
	f.formats = f.formats[1:]
	return r.cfg, r.info, r.err
}

// leaked reports buffers checked out but never returned.
func (f *fakeSource) leaked() []uint32 {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []uint32
	for i := range f.dequeued {
		out = append(out, i)
	}
	return out
}

var (
	streamCfg  = DeviceConfig{Format: 'Y' | 'U'<<8 | 'Y'<<16 | 'V'<<24, Width: 2, Height: 2, FPS: Frac{30, 1}}
	streamInfo = BufferInfo{BufferSize: 8, ImageStride: 4}
)

func newTestStream(f *fakeSource, sc StreamConfig) *Stream {
	return newStream(f, sc, streamCfg, streamInfo)
}

func waitFin(t *testing.T, s *Stream) {
	t.Helper()
	select {
	case <-s.fin:
	case <-time.After(5 * time.Second):
		t.Fatal("capture goroutine did not exit")
	}
}

func monoFrames(n int) []fakeFrame {
	var out []fakeFrame
	for i := 0; i < n; i++ {
		out = append(out, fakeFrame{
			index: uint32(i % 2),
			data:  []byte{byte(i), byte(i + 1)},
			seq:   uint32(i),
			ts:    v4l_timeval{sec: int64(i + 1)},
			flags: v4l_bufFlagTimestampMonotonic,
		})
	}
	return out
}

func TestStreamTimestamps(t *testing.T) {
	fs := newFakeSource(2, 8)
	fs.feed(
		fakeFrame{index: 0, data: []byte{1}, seq: 10,
			ts: v4l_timeval{sec: 1, usec: 500000}, flags: v4l_bufFlagTimestampMonotonic},
		fakeFrame{index: 1, data: []byte{2}, seq: 11,
			ts: v4l_timeval{sec: 2}, flags: v4l_bufFlagTimestampCopy},
		// Zero timestamp: must fall back to the local clock.
		fakeFrame{index: 0, data: []byte{3}, seq: 12,
			ts: v4l_timeval{}, flags: v4l_bufFlagTimestampMonotonic},
		// Unknown timestamp clock: must fall back even if nonzero.
		fakeFrame{index: 1, data: []byte{4}, seq: 13,
			ts: v4l_timeval{sec: 9, usec: 9}, flags: v4l_bufFlagTimestampUnknown},
	)
	fs.finish(io.EOF)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 4})

	f, err := s.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if f.Timestamp != 1500*time.Millisecond || f.TimestampFallback {
		t.Errorf("frame 0: got %v (fallback=%v), want 1.5s (fallback=false)",
			f.Timestamp, f.TimestampFallback)
	}
	if f.SeqNum != 10 {
		t.Errorf("frame 0: got seq %d, want 10", f.SeqNum)
	}
	if f.Config != streamCfg || f.Info != streamInfo {
		t.Errorf("frame 0: got format %+v/%+v, want %+v/%+v",
			f.Config, f.Info, streamCfg, streamInfo)
	}

	f, err = s.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if f.Timestamp != 2*time.Second || f.TimestampFallback {
		t.Errorf("frame 1: got %v (fallback=%v), want 2s (fallback=false)",
			f.Timestamp, f.TimestampFallback)
	}

	for i, want := range []uint32{12, 13} {
		f, err = s.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if !f.TimestampFallback {
			t.Errorf("frame %d: expected timestamp fallback", i+2)
		}
		if f.Timestamp < 0 || f.Timestamp > time.Minute {
			t.Errorf("frame %d: fallback timestamp %v out of range", i+2, f.Timestamp)
		}
		if f.SeqNum != want {
			t.Errorf("frame %d: got seq %d, want %d", i+2, f.SeqNum, want)
		}
	}

	if _, err := s.Next(context.Background()); err != io.EOF {
		t.Errorf("got %v, want io.EOF", err)
	}
	if leaked := fs.leaked(); len(leaked) != 0 {
		t.Errorf("leaked buffers: %v", leaked)
	}
}

func TestStreamDropOldest(t *testing.T) {
	fs := newFakeSource(2, 8)
	fs.feed(monoFrames(5)...)
	fs.finish(io.EOF)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 2, Policy: DropOldest})
	waitFin(t, s)

	// The consumer was not reading while the producer ran: only the two
	// most recent frames may survive.
	for i, want := range []uint32{3, 4} {
		f, err := s.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if f.SeqNum != want {
			t.Errorf("frame %d: got seq %d, want %d", i, f.SeqNum, want)
		}
	}
	if _, err := s.Next(context.Background()); err != io.EOF {
		t.Errorf("got %v, want io.EOF", err)
	}
	if n := s.Dropped(); n != 3 {
		t.Errorf("got %d dropped frames, want 3", n)
	}
	if leaked := fs.leaked(); len(leaked) != 0 {
		t.Errorf("leaked buffers: %v", leaked)
	}
}

func TestStreamBlockProducer(t *testing.T) {
	fs := newFakeSource(2, 8)
	fs.feed(monoFrames(3)...)
	fs.finish(io.EOF)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 1, Policy: BlockProducer})

	// A slow consumer must receive every frame, in order.
	for want := uint32(0); want < 3; want++ {
		time.Sleep(10 * time.Millisecond)
		f, err := s.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if f.SeqNum != want {
			t.Errorf("got seq %d, want %d", f.SeqNum, want)
		}
	}
	if _, err := s.Next(context.Background()); err != io.EOF {
		t.Errorf("got %v, want io.EOF", err)
	}
	if n := s.Dropped(); n != 0 {
		t.Errorf("got %d dropped frames, want 0", n)
	}
	if leaked := fs.leaked(); len(leaked) != 0 {
		t.Errorf("leaked buffers: %v", leaked)
	}
}

func TestStreamZeroQueueDrop(t *testing.T) {
	fs := newFakeSource(2, 8)
	fs.feed(monoFrames(3)...)
	fs.finish(io.EOF)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 0, Policy: DropOldest})
	waitFin(t, s)

	// Nobody was waiting for the frames: all of them must be dropped.
	if _, err := s.Next(context.Background()); err != io.EOF {
		t.Errorf("got %v, want io.EOF", err)
	}
	if n := s.Dropped(); n != 3 {
		t.Errorf("got %d dropped frames, want 3", n)
	}
	if leaked := fs.leaked(); len(leaked) != 0 {
		t.Errorf("leaked buffers: %v", leaked)
	}
}

func TestStreamZeroQueueBlock(t *testing.T) {
	fs := newFakeSource(2, 8)
	fs.feed(monoFrames(3)...)
	fs.finish(io.EOF)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 0, Policy: BlockProducer})

	// Unbuffered rendezvous: every frame is handed over directly.
	for want := uint32(0); want < 3; want++ {
		f, err := s.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if f.SeqNum != want {
			t.Errorf("got seq %d, want %d", f.SeqNum, want)
		}
	}
	if _, err := s.Next(context.Background()); err != io.EOF {
		t.Errorf("got %v, want io.EOF", err)
	}
	if n := s.Dropped(); n != 0 {
		t.Errorf("got %d dropped frames, want 0", n)
	}
}

func TestStreamOwnership(t *testing.T) {
	fs := newFakeSource(1, 4)
	// Two frames from the same device buffer: the second capture
	// overwrites the first one's memory.
	fs.feed(
		fakeFrame{index: 0, data: []byte{1, 2, 3, 4}, seq: 0,
			ts: v4l_timeval{sec: 1}, flags: v4l_bufFlagTimestampMonotonic},
		fakeFrame{index: 0, data: []byte{5, 6, 7, 8}, seq: 1,
			ts: v4l_timeval{sec: 2}, flags: v4l_bufFlagTimestampMonotonic},
	)
	fs.finish(io.EOF)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 2})
	waitFin(t, s)

	f0, err := s.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	f1, err := s.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// The device buffer has been returned, overwritten with 0xff, and
	// reused for the second frame; the first frame's data must be intact.
	for i, want := range []byte{1, 2, 3, 4} {
		if f0.Data[i] != want {
			t.Errorf("frame 0 data[%d]: got %d, want %d", i, f0.Data[i], want)
		}
	}
	for i, want := range []byte{5, 6, 7, 8} {
		if f1.Data[i] != want {
			t.Errorf("frame 1 data[%d]: got %d, want %d", i, f1.Data[i], want)
		}
	}
	if leaked := fs.leaked(); len(leaked) != 0 {
		t.Errorf("leaked buffers: %v", leaked)
	}
	if n := len(fs.requeued); n != 2 {
		t.Errorf("got %d requeued buffers, want 2", n)
	}
}

func TestStreamCancel(t *testing.T) {
	fs := newFakeSource(2, 8)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 1})
	defer func() {
		fs.finish(io.EOF)
		s.Close()
	}()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Next(ctx); err != context.Canceled {
		t.Errorf("got %v, want context.Canceled", err)
	}

	ctx, cancel = context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := s.Next(ctx); err != context.DeadlineExceeded {
		t.Errorf("got %v, want context.DeadlineExceeded", err)
	}

	// Cancellation must not disturb the stream itself.
	fs.feed(fakeFrame{index: 0, data: []byte{7}, seq: 0,
		ts: v4l_timeval{sec: 1}, flags: v4l_bufFlagTimestampMonotonic})
	f, err := s.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if f.SeqNum != 0 || len(f.Data) != 1 || f.Data[0] != 7 {
		t.Errorf("unexpected frame after cancel: %+v", f)
	}
}

func TestStreamUnplug(t *testing.T) {
	fs := newFakeSource(2, 8)
	fs.feed(fakeFrame{index: 0, data: []byte{1}, seq: 0,
		ts: v4l_timeval{sec: 1}, flags: v4l_bufFlagTimestampMonotonic})
	fs.finish(syscall.ENODEV)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 1})

	if _, err := s.Next(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Next(context.Background()); err != syscall.ENODEV {
		t.Errorf("got %v, want ENODEV", err)
	}
	if err := s.Err(); err != syscall.ENODEV {
		t.Errorf("Err(): got %v, want ENODEV", err)
	}
	if leaked := fs.leaked(); len(leaked) != 0 {
		t.Errorf("leaked buffers: %v", leaked)
	}
}

func TestStreamErrorFramesDropped(t *testing.T) {
	fs := newFakeSource(2, 8)
	fs.feed(
		fakeFrame{index: 0, data: []byte{0xde, 0xad}, seq: 0,
			ts: v4l_timeval{sec: 1}, flags: v4l_bufFlagError},
		fakeFrame{index: 1, data: []byte{1}, seq: 1,
			ts: v4l_timeval{sec: 2}, flags: v4l_bufFlagTimestampMonotonic},
	)
	fs.finish(io.EOF)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 2})

	f, err := s.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if f.SeqNum != 1 {
		t.Errorf("got seq %d, want 1 (corrupt frame not skipped)", f.SeqNum)
	}
	if n := s.Dropped(); n != 1 {
		t.Errorf("got %d dropped frames, want 1", n)
	}
	if leaked := fs.leaked(); len(leaked) != 0 {
		t.Errorf("leaked buffers: %v", leaked)
	}
	s.Close()
}

func TestStreamFormatChange(t *testing.T) {
	fs := newFakeSource(2, 16)
	newCfg := DeviceConfig{Format: 'M' | 'J'<<8 | 'P'<<16 | 'G'<<24, Width: 4, Height: 4, FPS: Frac{15, 1}}
	newInfo := BufferInfo{BufferSize: 32, ImageStride: 16}
	fs.formats = []formatReply{{cfg: newCfg, info: newInfo}}
	fs.feed(
		// Oversized frame: does not fit the format the stream started with.
		fakeFrame{index: 0, data: make([]byte, 12), seq: 0,
			ts: v4l_timeval{sec: 1}, flags: v4l_bufFlagTimestampMonotonic},
		fakeFrame{index: 1, data: []byte{1}, seq: 1,
			ts: v4l_timeval{sec: 2}, flags: v4l_bufFlagTimestampMonotonic},
	)
	fs.finish(io.EOF)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 2})

	f, err := s.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if f.Config != newCfg || f.Info != newInfo {
		t.Errorf("frame 0: got format %+v/%+v, want %+v/%+v",
			f.Config, f.Info, newCfg, newInfo)
	}
	f, err = s.Next(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if f.Config != newCfg {
		t.Errorf("frame 1: got config %+v, want %+v", f.Config, newCfg)
	}
	s.Close()
}

func TestStreamConcurrentClose(t *testing.T) {
	fs := newFakeSource(2, 8)
	fs.feed(monoFrames(2)...)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 4})

	const consumers = 4
	results := make(chan error, consumers)
	for i := 0; i < consumers; i++ {
		go func() {
			for {
				_, err := s.Next(context.Background())
				if err != nil {
					results <- err
					return
				}
			}
		}()
	}

	// Let the consumers block on the drained queue, then stop the device
	// and close the stream concurrently.
	time.Sleep(50 * time.Millisecond)
	fs.finish(syscall.ENODEV)
	go s.Close()
	s.Close() // idempotent, concurrent with the other Close

	deadline := time.After(5 * time.Second)
	for i := 0; i < consumers; i++ {
		select {
		case err := <-results:
			if err != ErrStreamClosed && err != syscall.ENODEV {
				t.Errorf("consumer got %v, want ErrStreamClosed or ENODEV", err)
			}
		case <-deadline:
			t.Fatal("consumers did not finish")
		}
	}
	waitFin(t, s)
	if leaked := fs.leaked(); len(leaked) != 0 {
		t.Errorf("leaked buffers: %v", leaked)
	}
}

func TestStreamCloseDrainsQueue(t *testing.T) {
	fs := newFakeSource(2, 8)
	fs.feed(monoFrames(2)...)
	s := newTestStream(fs, StreamConfig{QueueCapacity: 4})

	// Wait for both frames to reach the queue.
	for deadline := time.Now().Add(5 * time.Second); ; {
		if len(s.frames) == 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("frames did not reach the queue")
		}
		time.Sleep(time.Millisecond)
	}

	// Close first, then let the blocked dequeue fail: the stream must
	// report ErrStreamClosed, not the dequeue error.
	s.closeOnce.Do(func() { close(s.done) })
	fs.finish(io.EOF)
	s.Close() // waits for the capture goroutine; idempotent
	waitFin(t, s)

	// Frames queued before Close must still be delivered.
	for want := uint32(0); want < 2; want++ {
		f, err := s.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if f.SeqNum != want {
			t.Errorf("got seq %d, want %d", f.SeqNum, want)
		}
	}
	if _, err := s.Next(context.Background()); err != ErrStreamClosed {
		t.Errorf("got %v, want ErrStreamClosed", err)
	}
}

func TestDeviceNewStreamValidation(t *testing.T) {
	d := &Device{&device{}}
	if _, err := d.NewStream(StreamConfig{QueueCapacity: -1}); err == nil {
		t.Error("negative queue capacity should have failed")
	}
	if _, err := d.NewStream(StreamConfig{}); err == nil {
		t.Error("stream on a device that is not capturing should have failed")
	}
}

func TestV4lBufferTimestampRoundTrip(t *testing.T) {
	in := v4l_buffer{
		index:     3,
		typ:       v4l_bufTypeVideoCapture,
		bytesused: 42,
		flags:     v4l_bufFlagTimestampMonotonic,
		timestamp: v4l_timeval{sec: 123, usec: 456789},
		sequence:  7,
		memory:    v4l_memoryMmap,
	}
	buf := make([]uint64, (in.size()+7)/8)
	p := unsafe.Pointer(&buf[0])
	in.put(p)
	var out v4l_buffer
	out.get(p)
	if out.timestamp != in.timestamp {
		t.Errorf("got timestamp %+v, want %+v", out.timestamp, in.timestamp)
	}
	if out.index != in.index || out.bytesused != in.bytesused ||
		out.flags != in.flags || out.sequence != in.sequence {
		t.Errorf("round trip mismatch: got %+v, want %+v", out, in)
	}
}
