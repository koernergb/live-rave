import { test, expect, Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const url = process.env.PARITY_URL ?? "http://localhost:4175";

const logText = (page: Page) => page.locator("#status").textContent();

test("browser repeatedly (WASM) reproduces the Python reference", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });

  await page.goto(url);
  await expect(page).toHaveTitle(/RAVE-Live/);
  await expect(page.locator("#status")).toContainText(
    "models loaded: v2-live · seed 0",
    { timeout: 60_000 },
  );

  await page.click("#parity");
  await expect
    .poll(async () => await logText(page), { timeout: 150_000 })
    .toContain("browser parity: ");

  const status = (await logText(page)) ?? "";
  expect(status).toContain("PASS");
  const m = status.match(/maxerr=([0-9.e-]+)/);
  expect(m).not.toBeNull();
  expect(parseFloat(m![1])).toBeLessThan(1e-4);
  expect(status).toContain("bad=0");
  expect(errors).toEqual([]);
});

test("file mode runs end-to-end and produces a WAV", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));

  const src = resolve(root, "test", "input.wav");
  await page.goto(url);
  await page.selectOption("#model-select", "organ-archive-b2048");
  await expect(page.locator("#status")).toContainText(
    "models loaded: Organ Archive · trained (IIL, offline)",
    { timeout: 60_000 },
  );
  await expect(page.locator("#rt-start")).toBeDisabled();
  await page.setInputFiles("#file", src);
  await page.click("#run");

  const dlPromise = page.waitForEvent("download", { timeout: 60_000 });
  await expect
    .poll(async () => await logText(page), { timeout: 60_000 })
    .toContain("done.");
  await page.click("#download");

  const dl = await dlPromise;
  const out = await dl.path();
  expect(out).toBeTruthy();
  const bytes = readFileSync(out!);
  // RIFF/WAVE + PCM16 mono header sanity
  expect(bytes.subarray(0, 4).toString("ascii")).toBe("RIFF");
  expect(bytes.subarray(8, 12).toString("ascii")).toBe("WAVE");
  const fileSize = bytes.readUInt32LE(4) + 8;
  expect(bytes.byteLength).toBe(fileSize);
  expect(errors).toEqual([]);
});
