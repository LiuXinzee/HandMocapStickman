import { describe, expect, it } from "vitest";
import { stabilizeHandSurface } from "./handSurfaceStabilizer";

describe("stabilizeHandSurface", () => {
  it("publishes the first confident surface without an unknown delay", () => {
    expect(stabilizeHandSurface(undefined, "palm", 0.8)).toEqual({
      score: 0.8,
      surface: "palm",
    });
  });

  it("holds the last surface through unknown and one opposite frame", () => {
    const palm = stabilizeHandSurface(undefined, "palm", 0.8);
    const unknown = stabilizeHandSurface(palm, "unknown", 0);
    const oneBackFrame = stabilizeHandSurface(unknown, "back", 0.8);

    expect(unknown).toEqual(palm);
    expect(oneBackFrame.surface).toBe("palm");
  });

  it("switches only after sustained opposite evidence", () => {
    let state = stabilizeHandSurface(undefined, "palm", 0.8);
    for (let frame = 0; frame < 8; frame++) {
      state = stabilizeHandSurface(state, "back", 0.8);
    }

    expect(state.surface).toBe("back");
    expect(state.score).toBeLessThan(-0.28);
  });
});
