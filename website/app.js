// voxanim website: light on the viewer's machine and connection.
//
// * Every clip is a still (a lazy-loaded poster) until it is on screen; then
//   a 480p preview plays, at most PLAYING at a time, and the <video> element
//   is removed again when the clip leaves, so a weak device decodes no more
//   than two streams and loads nothing it does not show.
// * Previews are off when the browser asks for reduced motion or data, on a
//   slow connection, or by the "Previews" button; the choice is remembered.
// * Dark mode follows the system unless the "Dark mode" button chooses.
// * The player plays the full 960×540 file.
//
// The page works without this script: posters, links and downloads.
"use strict";

const PLAYING = 2;                      // previews playing at once

const root = document.documentElement;
const player = document.getElementById("player");
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

// ---------------------------------------------------------------- previews on or off

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const reduceData = matchMedia("(prefers-reduced-data: reduce)");
const connection = navigator.connection;
const slow = Boolean(connection && (connection.saveData ||
                                    /(^|-)2g$|^3g$/.test(connection.effectiveType || "")));
const previewsButton = document.getElementById("previews");
let previews = store.get("previews") ? store.get("previews") === "on"
                                     : !(reduceMotion.matches || reduceData.matches || slow);

function showPreviewsChoice() {
  previewsButton.textContent = previews ? "Previews on" : "Previews off";
  previewsButton.setAttribute("aria-pressed", String(previews));
  previewsButton.title = previews ? "Clips play as small previews while on screen"
                                  : "Clips stay still; open one to play it";
}

showPreviewsChoice();
previewsButton.addEventListener("click", () => {
  previews = !previews;
  store.set("previews", previews ? "on" : "off");
  showPreviewsChoice();
  schedule();
});
reduceMotion.addEventListener("change", () => {
  if (!store.get("previews")) { previews = !reduceMotion.matches; showPreviewsChoice(); schedule(); }
});

// ---------------------------------------------------------------- the preview videos
// A frame holds a poster <img>; its <video> exists only while it plays.

const frames = [...document.querySelectorAll(".frame[data-preview]")];
const visible = [];                     // frames on screen, the most recent last

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

function schedule() {
  const active = (previews && !player.open) ? visible.slice(-PLAYING) : [];
  for (const frame of frames) {
    if (!active.includes(frame)) { drop(frame); continue; }
    const v = videoOf(frame);
    if (!v.getAttribute("src")) v.src = frame.dataset.preview;
    v.play().catch(() => {});           // autoplay refused: the poster stays
  }
}

const onScreen = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const i = visible.indexOf(e.target);
    if (i >= 0) visible.splice(i, 1);
    if (e.isIntersecting) visible.push(e.target);
  }
  schedule();
}, { threshold: 0.5 });
frames.forEach((f) => onScreen.observe(f));

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
  big.src = clip.mp4;                   // the full file: the previews are for the grid
  if (!player.open) player.showModal();
  schedule();                           // the previews rest while the player is open
  if (!reduceMotion.matches) big.play().catch(() => {});
  history.replaceState(null, "", "#" + clip.name);
}

player.addEventListener("close", () => {
  big.pause();
  big.removeAttribute("src");
  big.load();
  history.replaceState(null, "", location.pathname + location.search);
  schedule();
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
