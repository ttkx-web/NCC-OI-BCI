export const percent = (value: number | null | undefined, digits = 1) =>
  value == null ? "—" : `${(value * 100).toFixed(digits)}%`;

export const milliseconds = (value: number | null | undefined) =>
  value == null ? "—" : `${value.toFixed(1)} ms`;

