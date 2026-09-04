# 7adethni — website

Landing page, rebuilt clean from the DesignCode v2 mock (`7adethni_ Tunisian Arabizi Extensionn/7adethni v2.dc.html`).
Same palette (oklch dark/light), fonts (Space Grotesk · Hanken Grotesk · IBM Plex Sans Arabic · Space Mono),
copy and the live **transform demo** — but as plain HTML/CSS/JS (no React/DesignCode runtime), so it deploys anywhere.

## Files
- `index.html` — nav, hero/Overview, transform demo, Onboarding, form factors (Popup/Side panel/Inline), Tones, CTA.
- `style.css` — design tokens + components, dark default with a `[data-theme="light"]` flip.
- `app.js` — theme toggle, scroll-spy nav, tone chips, and the transform demo.

## The transform demo
By default it runs **offline** with the same mock dictionary the mock used (so the page is alive with no server).
To wire it to the **real model**, set `BACKEND_URL` at the top of `app.js` to your deployed gateway:

```js
const BACKEND_URL = 'https://api.your-domain.tn'; // backend/app.py
```

It POSTs `{ feature:'translate', tone, text, device_id }` to `BACKEND_URL/generate` and reads `output`
— exactly what `backend/app.py` returns. If the call fails it silently falls back to the offline demo.

## Run locally
Just open `index.html`, or:
```bash
cd website && python -m http.server 5173   # http://localhost:5173
```

## Deploy
Static — drop the folder on GitHub Pages / Netlify / Vercel / Cloudflare Pages. No build step.

## Brand consistency
The browser extension (`../extension`) shares the same tokens/fonts, so the popup and the site look like one product.
