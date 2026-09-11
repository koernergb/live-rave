import { test, expect } from "@playwright/test";
const url = process.env.LIVE_URL ?? "http://localhost:4175";

// M4 gate: the latent-engine UI is present and honest — picker lists all
// catalog models with byte sizes, latent sliders drive real dims, all routing
// / EQ / noise controls exist, and a short realtime run at max slack posts
// latent scope frames with zero underruns.
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

test("M4 live UI: picker, latent controls, EQ net, realtime scope (0 underruns)", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });

  await page.goto(url);
  await expect(page).toHaveTitle(/RAVE-Live/);
  const coi = await page.evaluate(() => self.crossOriginIsolated === true);
  expect(coi).toBe(true);

  // Catalog picker: >=4 lazy-load options with MB sizes; default model boots.
  const options = page.locator("#model-select option");
  await expect(options).toHaveCount(5);
  await expect(page.locator("#model-info")).toContainText("MB");
  const status = page.locator("#status");
  await expect(status).toContainText("models loaded: v2-live · seed 0");

  // Latent controls: one bias + one scale per dim (8 dims).
  const lat = page.locator("#lat-row input[type='range']");
  await expect(lat).toHaveCount(16);

  // Tune the net: latent, noise, input gain, 3-band EQ, wet, slack.
  await page.locator("#noise").fill("150");
  await expect(page.locator("#noise-v")).toHaveText("1.50");
  await page.locator("#gain").fill("200");
  await expect(page.locator("#gain-v")).toHaveText("2.00");
  await page.locator("#eq-low").fill("-6");
  await expect(page.locator("#eq-low-v")).toHaveText("-6 dB");
  await page.locator("#eq-mid").fill("4");
  await expect(page.locator("#eq-mid-v")).toHaveText("4 dB");
  await page.locator("#eq-high").fill("6");
  await expect(page.locator("#eq-high-v")).toHaveText("6 dB");
  await page.locator("#wet").fill("60");
  await expect(page.locator("#wet-v")).toHaveText("60% wet");
  await page.locator("#slack").fill("10");
  await expect(page.locator("#slack-v")).toHaveText("10 blocks");
  await page.locator("#lat-row input[type='range']").nth(0).fill("1.5");
  await page.locator("#lat-row input[type='range']").nth(15).fill("0.5");

  // Realtime: fake mic → RAVE → speakers; scope canvas must receive frames.
  await page.click("#rt-start");
  await expect
    .poll(
      async () => Number((await page.locator("#scope-n").textContent()) ?? 0),
      { timeout: 90_000 },
    )
    .toBeGreaterThan(0);
  await expect
    .poll(
      async () =>
        Number((await page.locator("#rt-blocks").textContent()) ?? 0),
      { timeout: 30_000 },
    )
    .toBeGreaterThan(0);

  await page.waitForTimeout(10_000);

  const blocks = Number(await page.locator("#rt-blocks").textContent());
  const underruns = Number(await page.locator("#rt-underruns").textContent());
  const avg = Number.parseFloat(
    (await page.locator("#rt-avg").textContent()) ?? "-1",
  );
  const scope = Number(await page.locator("#scope-n").textContent());

  expect(blocks).toBeGreaterThan(0);
  expect(underruns).toBe(0);
  expect(avg).toBeGreaterThan(0);
  expect(avg).toBeLessThan(60);
  expect(scope).toBeGreaterThan(0);
  expect(errors).toEqual([]);
  await page.click("#rt-stop");

  // Model switch lazy-loads a different checkpoint (byte sizes update) and
  // tears the realtime graph down so a stale model never runs.
  await page.selectOption("#model-select", "v2-s1");
  await expect(status).toContainText("models loaded: v2 · seed 1", {
    timeout: 60_000,
  });
  await expect(page.locator("#rt-blocks")).toHaveText("0");
});