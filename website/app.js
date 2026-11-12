// voxanim website: every clip animates, cheaply.
//
// * A clip's frame holds its poster <img>.  While the clip is near the screen
//   a <video> plays on top: the thumbnail (the first half of the animation at
//   320 px and 12 frames a second, with a 2 s hold before it loops), under
//   100 KB and cheap to decode.  Off screen, the element goes.
// * While a clip is hovered or focused it plays its full-rate 480 px preview
//   instead; clicking opens the player with the full 960×540 file.
// * Under prefers-reduced-motion the thumbnails stay still (hover still
//   plays).  Dark mode follows the system unless the button chooses.
//
// The page works without this script: posters, links and downloads.
"use strict";

const root = document.documentElement;
const player = document.getElementById("player");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const store = {                         // per-viewer preferences; storage may be unavailable
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private window, blocked storage */ } },
};

// ---------------------------------------------------------------- theme

const themeButton = document.getElementById("theme");
const systemDark = matchMedia("(prefers-color-scheme: dark)");

function isDark() {
  return root.dataset.theme ? root.dataset.theme === "dark" : systemDark.matches;
}

function applyTheme(choice) {
  if (choice) root.dataset.theme = choice;
  else delete root.dataset.theme;
  themeButton.textContent = isDark() ? "Light mode" : "Dark mode";
  themeButton.setAttribute("aria-pressed", String(isDark()));
}

applyTheme(store.get("theme"));
themeButton.addEventListener("click", () => {
  const choice = isDark() ? "light" : "dark";
  store.set("theme", choice);
  applyTheme(choice);
});
systemDark.addEventListener("change", () => applyTheme(root.dataset.theme || null));

// ---------------------------------------------------------------- the clips
// Each frame plays one source at a time: its thumbnail, or its preview while
// highlighted.  Frames beyond a screen's distance hold no <video> at all.

const frames = [...document.querySelectorAll(".frame[data-thumb], .frame[data-preview]")];
const near = new Set();                 // frames within reach of the screen
const highlighted = new Set();          // frames hovered or focused

function videoOf(frame) {
  let v = frame.querySelector("video");
  if (!v) {
    v = document.createElement("video");
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    v.preload = "none";
    v.setAttribute("aria-hidden", "true");
    v.addEventListener("playing", () => v.classList.add("on"));
    frame.appendChild(v);
  }
  return v;
}

function drop(frame) {
  const v = frame.querySelector("video");
  if (!v) return;
  v.pause();
  v.removeAttribute("src");
  v.load();                             // releases the decoder
  v.remove();
}

function wanted(frame) {
  if (player.open || !near.has(frame)) return null;
  if (highlighted.has(frame)) return frame.dataset.preview || frame.dataset.thumb;
  if (reduceMotion.matches) return null;
  return frame.dataset.thumb || frame.dataset.preview;   // the hero has only a preview
}

function update(frame) {
  const src = wanted(frame);
  if (!src) { drop(frame); return; }
  const v = videoOf(frame);
  if (v.getAttribute("src") !== src) {
    v.classList.remove("on");
    v.src = src;
  }
  v.play().catch(() => {});             // autoplay refused: the poster stays
}

function updateAll() { frames.forEach(update); }

const reach = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) near.add(e.target);
    else near.delete(e.target);
    update(e.target);
  }
}, { rootMargin: "100% 0px" });        // a screen ahead and behind
frames.forEach((f) => reach.observe(f));

for (const frame of frames) {
  const target = frame.closest("a") || frame;
  const on = () => { highlighted.add(frame); update(frame); };
  const off = () => { highlighted.delete(frame); update(frame); };
  target.addEventListener("mouseenter", on);
  target.addEventListener("mouseleave", off);
  target.addEventListener("focusin", on);
  target.addEventListener("focusout", off);
}
reduceMotion.addEventListener("change", updateAll);

// ---------------------------------------------------------------- step bar
// Like the step bar in the animations: sections above in ink, the current
// one in blue, the rest grey.

const steps = [...document.querySelectorAll(".steps a")];
const sections = steps.map((a) => document.querySelector(a.hash));

function markStep(current) {
  const i = sections.indexOf(current);
  steps.forEach((a, j) => {
    a.classList.toggle("done", j < i);
    if (j === i) a.setAttribute("aria-current", "true");
    else a.removeAttribute("aria-current");
  });
  if (i >= 0) steps[i].scrollIntoView({ block: "nearest", inline: "nearest" });
}

const spy = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) markStep(e.target);
}, { rootMargin: "-30% 0px -65% 0px" });
sections.forEach((s) => s && spy.observe(s));

// ---------------------------------------------------------------- player

const big = player.querySelector("video");
const field = (sel) => player.querySelector(sel);

function length(seconds) {
  const m = Math.floor(seconds / 60), s = seconds % 60;
  const sec = (n) => n + (n === 1 ? " second" : " seconds");
  return m ? m + (m === 1 ? " minute " : " minutes ") + sec(s) : sec(s);
}

function openClip(clip) {
  field("#player-title").textContent = clip.title;
  field(".player-about").textContent = clip.about;
  field(".player-length").textContent = length(clip.seconds) + ", looping";
  for (const a of player.querySelectorAll("[data-file]")) a.href = clip[a.dataset.file];
  field(".player-cmd code").textContent = "python3 -m voxanim render " + clip.name;
  field(".copy").textContent = "Copy";
  big.poster = clip.poster;
  big.src = clip.mp4;                   // the full file
  if (!player.open) player.showModal();
  updateAll();                          // the grid rests while the player is open
  if (!reduceMotion.matches) big.play().catch(() => {});
  history.replaceState(null, "", "#" + clip.name);
}

player.addEventListener("close", () => {
  big.pause();
  big.removeAttribute("src");
  big.load();
  history.replaceState(null, "", location.pathname + location.search);
  updateAll();
});

// a click on the backdrop (outside the dialog's box) closes it
player.addEventListener("click", (e) => {
  const r = player.getBoundingClientRect();
  const inside = e.clientX >= r.left && e.clientX <= r.right &&
                 e.clientY >= r.top && e.clientY <= r.bottom;
  if (!inside) player.close();
});

document.addEventListener("click", (e) => {
  const link = e.target.closest("a[data-clip]");
  if (!link || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
  e.preventDefault();
  openClip(JSON.parse(link.dataset.clip));
});

field(".copy").addEventListener("click", async (e) => {
  try {
    await navigator.clipboard.writeText(field(".player-cmd code").textContent);
    e.target.textContent = "Copied";
  } catch {
    e.target.textContent = "Select and copy";
  }
});

// a link to #<animation> opens it (a link to #<section> just scrolls)
function openFromHash() {
  const target = location.hash && document.getElementById(location.hash.slice(1));
  const link = target && target.matches(".clip") && target.querySelector("a[data-clip]");
  if (link) openClip(JSON.parse(link.dataset.clip));
}
openFromHash();
window.addEventListener("hashchange", openFromHash);
