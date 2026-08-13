import { describe, expect, it } from "vitest";
import {
  GloveParser,
  HEADER,
  PACKET_TYPE_1,
  PACKET_TYPE_2,
  TYPE2_LEN_LEGACY,
  TYPE2_LEN_WITH_ACC,
  type GloveFrame,
  type GloveType2Length,
} from "./gloveProtocol";

const SENSOR_TYPE = 0x01;
const QUATERNION: [number, number, number, number] = [0.5, 0.5, 0.5, 0.5];
const ACCELERATION: [number, number, number] = [1.25, -2.5, 9.75];
const ATTITUDE: [number, number, number] = [45, -12.5, 3.25];

function concatBytes(...parts: Uint8Array[]): Uint8Array {
  const result = new Uint8Array(
    parts.reduce((length, part) => length + part.length, 0)
  );
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

function sensorBytes(seed: number, length = 128): Uint8Array {
  return Uint8Array.from({ length }, (_, index) => (seed + index * 17) & 0xff);
}

function writeFloat32s(
  target: Uint8Array,
  offset: number,
  values: readonly number[]
) {
  const view = new DataView(
    target.buffer,
    target.byteOffset,
    target.byteLength
  );
  values.forEach((value, index) => {
    view.setFloat32(offset + index * 4, value, true);
  });
}

function packet(
  packetOrder: number,
  data: Uint8Array,
  sensorType = SENSOR_TYPE
): Uint8Array {
  return concatBytes(HEADER, Uint8Array.of(packetOrder, sensorType), data);
}

function packet1(seed: number, sensorType = SENSOR_TYPE): Uint8Array {
  return packet(PACKET_TYPE_1, sensorBytes(seed), sensorType);
}

function packet2(
  seed: number,
  dataLength: typeof TYPE2_LEN_LEGACY | typeof TYPE2_LEN_WITH_ACC,
  sensorType = SENSOR_TYPE
): Uint8Array {
  const data = new Uint8Array(dataLength);
  data.set(sensorBytes(seed), 0);
  writeFloat32s(data, 128, QUATERNION);
  if (dataLength === TYPE2_LEN_WITH_ACC) {
    writeFloat32s(data, 144, ACCELERATION);
    writeFloat32s(data, 156, ATTITUDE);
  }
  return packet(PACKET_TYPE_2, data, sensorType);
}

function expectedSensorData(firstSeed: number, secondSeed: number): number[] {
  return Array.from(
    concatBytes(sensorBytes(firstSeed), sensorBytes(secondSeed))
  );
}

function createParser(
  options: { type2Len?: GloveType2Length; now?: () => number } = {}
) {
  const frames: GloveFrame[] = [];
  const parser = new GloveParser({
    ...options,
    onFrame: frame => frames.push(frame),
  });
  return { parser, frames };
}

function expectBaseFrame(
  frame: GloveFrame,
  firstSeed: number,
  secondSeed: number
) {
  expect(frame.sensor_data).toEqual(expectedSensorData(firstSeed, secondSeed));
  expect(frame.quaternion).toEqual(QUATERNION);
}

describe("GloveParser adaptive type-2 packet length", () => {
  it("auto-detects a 144-byte type-2 body and leaves extension fields null", () => {
    const { parser, frames } = createParser();

    parser.push(
      concatBytes(packet1(1), packet2(101, TYPE2_LEN_LEGACY), packet1(2))
    );

    expect(frames).toHaveLength(1);
    expectBaseFrame(frames[0], 1, 101);
    expect(frames[0].acceleration).toBeNull();
    expect(frames[0].attitude).toBeNull();
  });

  it("auto-detects a 168-byte type-2 body and parses its IMU extension", () => {
    const { parser, frames } = createParser();

    parser.push(
      concatBytes(packet1(3), packet2(103, TYPE2_LEN_WITH_ACC), packet1(4))
    );

    expect(frames).toHaveLength(1);
    expectBaseFrame(frames[0], 3, 103);
    expect(frames[0].acceleration).toEqual(ACCELERATION);
    expect(frames[0].attitude).toEqual(ATTITUDE);
  });

  it("waits for all four bytes of the following header before choosing 144", () => {
    const { parser, frames } = createParser();
    parser.push(packet1(5));
    parser.push(
      concatBytes(packet2(105, TYPE2_LEN_LEGACY), HEADER.slice(0, 3))
    );

    expect(frames).toHaveLength(0);

    parser.push(HEADER.slice(3));

    expect(frames).toHaveLength(1);
    expectBaseFrame(frames[0], 5, 105);
  });

  it("handles 144-byte and 168-byte type-2 bodies in the same stream", () => {
    const { parser, frames } = createParser();

    parser.push(
      concatBytes(
        packet1(6),
        packet2(106, TYPE2_LEN_LEGACY),
        packet1(7),
        packet2(107, TYPE2_LEN_WITH_ACC),
        packet1(8)
      )
    );

    expect(frames).toHaveLength(2);
    expectBaseFrame(frames[0], 6, 106);
    expect(frames[0].acceleration).toBeNull();
    expect(frames[0].attitude).toBeNull();
    expectBaseFrame(frames[1], 7, 107);
    expect(frames[1].acceleration).toEqual(ACCELERATION);
    expect(frames[1].attitude).toEqual(ATTITUDE);
    expect(frames[1].timestamp).toBeGreaterThan(frames[0].timestamp);
  });

  it("assigns strictly increasing timestamps to multiple frames in one read chunk", () => {
    const { parser, frames } = createParser();
    parser.push(
      concatBytes(
        packet1(20),
        packet2(120, TYPE2_LEN_LEGACY),
        packet1(21),
        packet2(121, TYPE2_LEN_LEGACY),
        packet1(22)
      )
    );

    expect(frames).toHaveLength(2);
    expect(frames[1].timestamp).toBeGreaterThan(frames[0].timestamp);
  });

  it("does not push rapid consecutive reads a full frame into the future", () => {
    const receivedAt = 1000;
    const { parser, frames } = createParser({ now: () => receivedAt });

    parser.push(
      concatBytes(packet1(30), packet2(130, TYPE2_LEN_LEGACY), packet1(31))
    );
    parser.push(concatBytes(packet2(131, TYPE2_LEN_LEGACY), packet1(32)));

    expect(frames).toHaveLength(2);
    expect(frames[1].timestamp).toBeGreaterThan(frames[0].timestamp);
    expect(frames[1].timestamp).toBeLessThanOrEqual(receivedAt + 0.01);
  });
});

describe("GloveParser stream recovery", () => {
  it("keeps alignment across a sustained legacy stream and irregular chunks", () => {
    const { parser, frames } = createParser();
    const parts: Uint8Array[] = [];
    for (let index = 0; index < 128; index++) {
      parts.push(packet1(index), packet2(128 + index, TYPE2_LEN_LEGACY));
    }
    parts.push(packet1(250));
    const stream = concatBytes(...parts);
    const chunkSizes = [1, 7, 149, 4096, 37];
    let offset = 0;
    let chunkIndex = 0;
    while (offset < stream.length) {
      const end = Math.min(
        stream.length,
        offset + chunkSizes[chunkIndex % chunkSizes.length]
      );
      parser.push(stream.subarray(offset, end));
      offset = end;
      chunkIndex++;
    }

    expect(frames).toHaveLength(128);
    expect(parser.count).toBe(128);
    expect(parser.pendingByteCount).toBe(0);
    expectBaseFrame(frames[0], 0, 128);
    expectBaseFrame(frames[127], 127, 255);
    expect(
      frames.every(
        (frame, index) =>
          index === 0 || frame.timestamp > frames[index - 1].timestamp
      )
    ).toBe(true);
  });

  it("recovers when capture starts at a type-2 packet", () => {
    const { parser, frames } = createParser();

    parser.push(
      concatBytes(
        packet2(108, TYPE2_LEN_LEGACY),
        packet1(9),
        packet2(109, TYPE2_LEN_LEGACY),
        packet1(10)
      )
    );

    expect(frames).toHaveLength(1);
    expectBaseFrame(frames[0], 9, 109);
  });

  it("skips garbage and invalid packet orders, then resynchronizes", () => {
    const { parser, frames } = createParser();
    const garbage = Uint8Array.of(0x00, 0xaa, 0x55, 0x03, 0x18, 0xfe);
    const invalidPacket = concatBytes(
      HEADER,
      Uint8Array.of(0x7f, SENSOR_TYPE),
      Uint8Array.of(0xde, 0xad, 0xbe, 0xef)
    );

    parser.push(
      concatBytes(
        garbage,
        invalidPacket,
        packet1(11),
        packet2(111, TYPE2_LEN_LEGACY),
        packet1(12)
      )
    );

    expect(frames).toHaveLength(1);
    expectBaseFrame(frames[0], 11, 111);
  });

  it("trims headerless input instead of growing the buffer indefinitely", () => {
    const { parser } = createParser();

    parser.push(new Uint8Array(64 * 1024).fill(0x7e));

    expect(parser.pendingByteCount).toBeLessThan(HEADER.length);
  });

  it("keeps packet-1 caches isolated between parser instances", () => {
    const first = createParser();
    const second = createParser();

    first.parser.push(packet1(13));
    second.parser.push(packet1(14));
    first.parser.push(concatBytes(packet2(113, TYPE2_LEN_LEGACY), HEADER));
    second.parser.push(concatBytes(packet2(114, TYPE2_LEN_LEGACY), HEADER));

    expect(first.frames).toHaveLength(1);
    expectBaseFrame(first.frames[0], 13, 113);
    expect(second.frames).toHaveLength(1);
    expectBaseFrame(second.frames[0], 14, 114);
  });
});

describe("GloveParser explicit type-2 packet length", () => {
  it("retains explicit 144-byte parsing without waiting for a next header", () => {
    const { parser, frames } = createParser({ type2Len: TYPE2_LEN_LEGACY });

    parser.push(concatBytes(packet1(15), packet2(115, TYPE2_LEN_LEGACY)));

    expect(frames).toHaveLength(1);
    expectBaseFrame(frames[0], 15, 115);
    expect(frames[0].acceleration).toBeNull();
    expect(frames[0].attitude).toBeNull();
  });

  it("retains explicit 168-byte parsing without waiting for a next header", () => {
    const { parser, frames } = createParser({
      type2Len: TYPE2_LEN_WITH_ACC,
    });

    parser.push(concatBytes(packet1(16), packet2(116, TYPE2_LEN_WITH_ACC)));

    expect(frames).toHaveLength(1);
    expectBaseFrame(frames[0], 16, 116);
    expect(frames[0].acceleration).toEqual(ACCELERATION);
    expect(frames[0].attitude).toEqual(ATTITUDE);
  });
});
