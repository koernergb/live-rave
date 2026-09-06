import { test, expect } from "@playwright/test";
const url = process.env.RT_URL ?? "http://localhost:4175";
const rtSeconds = Number(process.env.RT_SECONDS ?? 60);
// One block is stride samples @44.1kHz = 46.44ms. Average WASM turnaround is
// ~30ms; allow a generous ceiling but still require real-time feasibility.
const MAX_AVG_MS = 60;

test.use({
  launchOptions: {
    args: [
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
      "--autoplay-policy=no-user-gesture-required",
    ],
  },
  permissions: ["microphone"],
});

test(`runs ${rtSeconds}s with zero underruns (avg turnaround ${MAX_AVG_MS}ms)`, async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(m.text());
    });

    await page.goto(url);
    await expect(page).toHaveTitle(/RAVE-Live/);

    // SAB requires COOP/COEP (served by the COI middleware).
    const coi = await page.evaluate(() => self.crossOriginIsolated === true);
    expect(coi).toBe(true);

    await page.evaluate(() => {
      // kick off the loop-back; renderer promise resolutions drive nothing here
      (document.getElementById("rt-start") as HTMLButtonElement).click();
    });

    // First metric batch = models loaded + ≥16 blocks processed.
    await expect
      .poll(
        async () =>
          Number((await page.locator("#rt-blocks").textContent()) ?? 0),
        { timeout: 120_000 },
      )
      .toBeGreaterThan(0);

    // Run for the window, then assert the counters.
    await page.waitForTimeout(rtSeconds * 1000);
    await expect
      .poll(
        async () =>
          Number((await page.locator("#rt-blocks").textContent()) ?? 0),
        { timeout: 15_000 },
      )
      .toBeGreaterThan(0);

    const blocks = Number(await page.locator("#rt-blocks").textContent());
    const underruns = Number(
      await page.locator("#rt-underruns").textContent(),
    );
    const avg = Number.parseFloat(
      (await page.locator("#rt-avg").textContent()) ?? "NaN",
    );

    expect(blocks).toBeGreaterThan(0);
    expect(underruns).toBe(0);
    expect(avg).toBeGreaterThan(0);
    expect(avg).toBeLessThan(MAX_AVG_MS);
    expect(errors).toEqual([]);
  });