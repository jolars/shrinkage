const maximum = 2;
const offset = 0.001;
const logRange = Math.log1p(maximum / offset);

// The offset gives weak penalties room on the slider while retaining exact zero.
export const strengthAt = (position) =>
  position === 1 ? maximum : offset * Math.expm1(position * logRange);

export const strengthPosition = (strength) =>
  Math.log1p(strength / offset) / logRange;
