import queue
import math
import struct
import unittest

from glove_bridge.glove_ws_bridge import (
    HEADER,
    PACKET_TYPE_1,
    PACKET_TYPE_2,
    GloveSerialReader,
)


def make_packet(packet_order, sensor_type, data):
    return HEADER + bytes((packet_order, sensor_type)) + data


def make_packet_pair(sensor_type=1, extended=False, fill=0):
    packet1_data = bytes((fill + index) % 256 for index in range(128))
    packet2_sensors = bytes((fill + 128 + index) % 256 for index in range(128))
    quaternion = struct.pack('<4f', 1.0, 0.1, -0.2, 0.3)
    packet2_data = packet2_sensors + quaternion
    if extended:
        packet2_data += struct.pack('<3f', 1.25, -2.5, 3.75)
        packet2_data += struct.pack('<3f', 10.0, 20.0, 30.0)
    return (
        make_packet(PACKET_TYPE_1, sensor_type, packet1_data),
        make_packet(PACKET_TYPE_2, sensor_type, packet2_data),
        packet1_data + packet2_sensors,
    )


class GloveSerialReaderTests(unittest.TestCase):
    def test_adapts_to_mixed_144_and_168_byte_packet2(self):
        reader = GloveSerialReader('test')
        old_packet1, old_packet2, old_sensors = make_packet_pair(fill=3)
        ext_packet1, ext_packet2, ext_sensors = make_packet_pair(fill=41, extended=True)
        reader.buffer.extend(
            b'noise' + old_packet1 + old_packet2 + ext_packet1 + ext_packet2 + HEADER
        )

        reader._parse_buffer(1234.5)

        old_frame = reader.frame_queue.get_nowait()
        ext_frame = reader.frame_queue.get_nowait()
        with self.assertRaises(queue.Empty):
            reader.frame_queue.get_nowait()

        self.assertEqual(old_frame['sensor_data'], list(old_sensors))
        self.assertEqual(ext_frame['sensor_data'], list(ext_sensors))
        self.assertEqual(old_frame['timestamp'], 1234.5)
        self.assertEqual(ext_frame['timestamp'], 1234.5)
        self.assertIsNone(old_frame['acceleration'])
        self.assertIsNone(old_frame['attitude'])
        self.assertEqual(ext_frame['acceleration'], [1.25, -2.5, 3.75])
        self.assertEqual(ext_frame['attitude'], [10.0, 20.0, 30.0])
        self.assertAlmostEqual(ext_frame['quaternion'][1], 0.1, places=6)
        self.assertEqual(reader.buffer, HEADER)

    def test_repeated_mixed_packet2_formats_remain_synchronized(self):
        reader = GloveSerialReader('test')
        stream = bytearray()
        expected_extended = []
        for index in range(12):
            extended = index % 2 == 1
            packet1, packet2, _ = make_packet_pair(
                extended=extended,
                fill=index * 13,
            )
            stream.extend(packet1)
            stream.extend(packet2)
            expected_extended.append(extended)
        stream.extend(HEADER)
        reader.buffer.extend(stream)

        reader._parse_buffer(50.0)

        frames = [reader.frame_queue.get_nowait() for _ in range(12)]
        self.assertEqual(reader.frame_count, 12)
        self.assertEqual(
            [frame['acceleration'] is not None for frame in frames],
            expected_extended,
        )
        self.assertEqual(reader.buffer, HEADER)

    def test_waits_for_next_header_before_consuming_packet2(self):
        reader = GloveSerialReader('test')
        packet1, packet2, _ = make_packet_pair(extended=True)
        reader.buffer.extend(packet1 + packet2)

        reader._parse_buffer(10.0)

        self.assertEqual(reader.frame_count, 0)
        self.assertIn(1, reader.packet1_cache)
        self.assertEqual(reader.buffer, packet2)
        self.assertEqual(reader.error_count, 0)

        reader.buffer.extend(HEADER[:2])
        reader._parse_buffer(11.0)

        self.assertEqual(reader.frame_count, 0)
        self.assertEqual(reader.buffer, packet2 + HEADER[:2])

        reader.buffer.extend(HEADER[2:])
        reader._parse_buffer(12.0)

        frame = reader.frame_queue.get_nowait()
        self.assertEqual(frame['timestamp'], 12.0)
        self.assertEqual(reader.buffer, HEADER)

    def test_invalid_packet_order_resynchronizes_and_discards_noise(self):
        reader = GloveSerialReader('test')
        packet1, packet2, _ = make_packet_pair()
        invalid_packet_start = HEADER + bytes((0x7F, 1)) + b'bad'
        reader.buffer.extend(b'garbage' + invalid_packet_start + packet1 + packet2 + HEADER)

        reader._parse_buffer(1.0)

        self.assertEqual(reader.frame_count, 1)
        self.assertEqual(reader.error_count, 1)
        self.assertEqual(reader.buffer, HEADER)

        noise_reader = GloveSerialReader('test')
        noise_reader.buffer.extend(b'x' * 100 + HEADER[:3])
        noise_reader._parse_buffer(2.0)
        self.assertEqual(noise_reader.buffer, HEADER[:3])

    def test_bad_packet2_length_clears_stale_packet1(self):
        reader = GloveSerialReader('test')
        packet1, _, _ = make_packet_pair()
        reader.buffer.extend(packet1)
        reader._parse_buffer(1.0)
        self.assertIn(1, reader.packet1_cache)

        bad_packet2 = (
            HEADER
            + bytes((PACKET_TYPE_2, 1))
            + bytes(168)
            + b'not-a-header'
        )
        reader.buffer.extend(bad_packet2)
        reader._parse_buffer(2.0)

        self.assertNotIn(1, reader.packet1_cache)
        self.assertEqual(reader.error_count, 1)

    def test_packet1_cache_is_instance_local(self):
        left_reader = GloveSerialReader('left')
        right_reader = GloveSerialReader('right')
        left_packet1, _, _ = make_packet_pair(sensor_type=1, fill=10)
        right_packet1, _, _ = make_packet_pair(sensor_type=1, fill=90)

        left_reader.buffer.extend(left_packet1)
        right_reader.buffer.extend(right_packet1)
        left_reader._parse_buffer(1.0)
        right_reader._parse_buffer(2.0)

        self.assertIsNot(left_reader.packet1_cache, right_reader.packet1_cache)
        self.assertNotEqual(
            left_reader.packet1_cache[1],
            right_reader.packet1_cache[1],
        )

    def test_non_finite_extension_values_are_not_emitted(self):
        reader = GloveSerialReader('test')
        packet1, packet2, _ = make_packet_pair(extended=True)
        packet2 = bytearray(packet2)
        extension_start = len(HEADER) + 2 + 144
        packet2[extension_start:extension_start + 12] = struct.pack(
            '<3f', math.nan, 1.0, 2.0
        )
        reader.buffer.extend(packet1 + packet2 + HEADER)

        reader._parse_buffer(3.0)

        frame = reader.frame_queue.get_nowait()
        self.assertIsNone(frame['acceleration'])
        self.assertEqual(frame['attitude'], [10.0, 20.0, 30.0])


if __name__ == '__main__':
    unittest.main()
