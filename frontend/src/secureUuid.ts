const UNAVAILABLE = "浏览器暂不支持安全随机标识，无法上传或创建任务。请更换浏览器或使用 HTTPS。";

export function createSecureUuid(): string {
  const entropy = globalThis.crypto;
  if (typeof entropy?.randomUUID === "function") {
    try { return entropy.randomUUID(); } catch { /* Try the browser's other CSPRNG interface. */ }
  }
  if (typeof entropy?.getRandomValues === "function") {
    try {
      const bytes = entropy.getRandomValues(new Uint8Array(16));
      bytes[6] = (bytes[6] & 0x0f) | 0x40;
      bytes[8] = (bytes[8] & 0x3f) | 0x80;
      const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
      return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
    } catch { /* Fail before any upload or task submission when entropy is unavailable. */ }
  }
  throw new Error(UNAVAILABLE);
}
