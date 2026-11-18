// voxanim website: every clip animates, cheaply.
//
// * A clip's frame holds its poster <img>.  While the clip is near the screen
//   a <video> plays on top: the thumbnail (the first half of the animation at
//   480 px and 10 frames a second, with a 2 s hold before it loops), about
//   40 KB and cheap to decode.  Off screen, the element goes.
// * Posters and thumbnails come in light and dark; a frame shows the pair of
//   the page's theme (the light pair where an animation has no dark one) and
//   swaps when the theme changes.  Only the theme on show is ever fetched.
// * Clicking a clip opens the player, which streams the full 960×540 video,
//   light or dark, from the GitHub release: the site holds no full video.
// * Under prefers-reduced-motion the thumbnails stay still (hover still
//   plays).  Dark mode follows the system unless the button chooses.
//
// The page works without this script: posters and links.
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

// What follows the theme besides the stylesheet (the posters, the thumbnails, the open
// player): each part of the script below adds itself here.
const onTheme = [];
function setTheme(choice) {
  applyTheme(choice);
  onTheme.forEach((fn) => fn());
}

applyTheme(store.get("theme"));
themeButton.addEventListener("click", () => {
  const choice = isDark() ? "light" : "dark";
  store.set("theme", choice);
  setTheme(choice);
});
systemDark.addEventListener("change", () => setTheme(root.dataset.theme || null));

// ---------------------------------------------------------------- the clips
// Each frame plays its thumbnail in the page's theme.  Frames beyond a screen's
// distance hold no <video> at all.

const frames = [...document.querySelectorAll(".frame[data-thumb]")];
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
  if (reduceMotion.matches && !highlighted.has(frame)) return null;
  return (isDark() && frame.dataset.thumbDark) || frame.dataset.thumb;
}

// A poster is a <picture> whose dark <source> follows the system's colour scheme, which
// is right until the page's own button has chosen; from then on the choice is written
// into the <img> and the <source> goes.  A lazy <img> that has not been reached fetches
// nothing when its src changes, so a theme costs only the posters that are looked at.
function showPosters() {
  if (!root.dataset.theme) return;        // following the system: the <picture> does it
  for (const pic of document.querySelectorAll(".frame picture")) {
    const img = pic.querySelector("img");
    const source = pic.querySelector("source");
    if (source) {
      pic.dataset.light = img.getAttribute("src");
      source.remove();
    }
    const src = isDark() ? pic.dataset.dark : pic.dataset.light;
    if (img.getAttribute("src") !== src) img.src = src;
  }
}
showPosters();
onTheme.push(showPosters, () => updateAll());

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

const stepsNav = document.querySelector(".steps");
const steps = [...stepsNav.querySelectorAll("a")];
const sections = steps.map((a) => document.querySelector(a.hash));
const stepsToggle = stepsNav.querySelector(".steps-toggle");
const stepsCurrent = stepsNav.querySelector(".steps-current");

function markStep(current) {
  const i = sections.indexOf(current);
  steps.forEach((a, j) => {
    a.classList.toggle("done", j < i);
    if (j === i) a.setAttribute("aria-current", "true");
    else a.removeAttribute("aria-current");
  });
  if (i >= 0) stepsCurrent.textContent = (i + 1) + "\u2002" + steps[i].textContent;
  if (i >= 0) showStep(steps[i]);
}

// The bar is one line scrolled sideways: bring the current section's name to its middle.
// Only the bar moves; scrollIntoView would also move the page.
const stepsList = stepsNav.querySelector("ol");
function showStep(a) {
  if (stepsList.scrollWidth <= stepsList.clientWidth) return;
  const li = a.parentElement;
  const left = li.offsetLeft - stepsList.offsetLeft - (stepsList.clientWidth - li.offsetWidth) / 2;
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  stepsList.scrollTo({ left: Math.max(0, left), behavior: calm ? "auto" : "smooth" });
}

// Fast-forward and rewind, at the ends of the bar: on to the next section, or back to the
// top of this one and then to the one before, like a player's skip buttons. A section that
// opens a part of the course is reached at the part's heading, so the heading is seen.
const stops = sections.filter(Boolean).map((sec) => {
  const before = sec.previousElementSibling;
  return before && before.classList.contains("part") ? before : sec;
});
const skips = [...stepsNav.querySelectorAll(".steps-skip")];
const skipLine = () => stepsNav.offsetHeight + 12;      // just under the sticky bar
function stopAt() {                       // the stop the top of the window is in, or -1
  const line = skipLine() + 4;
  let k = -1;
  stops.forEach((el, j) => { if (el.getBoundingClientRect().top <= line) k = j; });
  return k;
}
function markSkips() {
  const k = stopAt();
  skips.forEach((b) => { b.disabled = +b.dataset.skip < 0 ? k < 0 : k >= stops.length - 1; });
}
function goTo(top) {
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  try {
    window.scrollTo({ top, behavior: calm ? "auto" : "smooth" });
  } catch (e) {
    window.scrollTo(0, top);              // a browser without scroll options
  }
}
skips.forEach((b) => b.addEventListener("click", () => {
  const k = stopAt();
  let to = k + 1;
  if (+b.dataset.skip < 0) {              // rewind: this section's top first
    const into = k >= 0 ? skipLine() - stops[k].getBoundingClientRect().top : 0;
    to = into > 120 ? k : k - 1;
  }
  if (to < 0) goTo(0);
  else if (to < stops.length) {
    goTo(Math.round(stops[to].getBoundingClientRect().top + window.scrollY - skipLine()));
  }
  setTimeout(markSkips, 700);             // once the page has come to rest
}));
window.addEventListener("scroll", markSkips, { passive: true });
markSkips();

// on narrow screens the list is behind a button naming the current section
function openSteps(open) {
  stepsNav.classList.toggle("open", open);
  stepsToggle.setAttribute("aria-expanded", String(open));
}
stepsToggle.addEventListener("click", () => openSteps(!stepsNav.classList.contains("open")));
steps.forEach((a) => a.addEventListener("click", () => openSteps(false)));

const spy = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) markStep(e.target);
}, { rootMargin: "-30% 0px -65% 0px" });
sections.forEach((s) => s && spy.observe(s));

// ---------------------------------------------------------------- player
// The full videos are files of the GitHub release, streamed from there: light as
// <media>.mp4, dark as <media>-dark.mp4.  The video's <source> states the type because
// the release serves every file as application/octet-stream.  The release sends no CORS
// header, so the video can be played and nothing else (no fetch, no canvas).

const big = player.querySelector("video");
const source = big.querySelector("source");
const failed = player.querySelector(".player-error");
const field = (sel) => player.querySelector(sel);
let clip = null;                        // the animation in the player
let showing = "light";                  // the theme of the video in the player
let retried = false;                    // the stream has been asked for a second time
let reached = 0;                        // the last position that played, in seconds

function length(seconds) {
  const m = Math.floor(seconds / 60), s = seconds % 60;
  const sec = (n) => n + (n === 1 ? " second" : " seconds");
  return m ? m + (m === 1 ? " minute " : " minutes ") + sec(s) : sec(s);
}

// Put the video of `theme` in the player (light where the animation has no dark one),
// starting at `at` seconds.
function show(theme, at, play) {
  showing = clip.video[theme] ? theme : "light";
  reached = at || 0;
  failed.hidden = true;
  big.hidden = false;
  big.poster = clip.poster[showing] || clip.poster.light;
  source.src = clip.video[showing];
  big.load();
  if (at) big.addEventListener("loadedmetadata", () => { big.currentTime = at; }, { once: true });
  if (play) big.play().catch(() => {});
  for (const b of player.querySelectorAll("[data-video-theme]")) {
    b.setAttribute("aria-pressed", String(b.dataset.videoTheme === showing));
    b.disabled = !clip.video[b.dataset.videoTheme];
  }
  field(".player-manifest").href = clip.manifest[showing];
  field(".player-cmd code").textContent = "python3 -m voxanim render " + clip.name +
    (showing === "dark" ? " --theme dark" : "");
  field(".copy").textContent = "Copy";
}

// The other theme of the same video, at the same moment.
function switchTo(theme) {
  if (!clip || !player.open || theme === showing || !clip.video[theme]) return;
  show(theme, big.currentTime || reached, !big.paused);
}

// GitHub answers each request for a release file with a redirect to a temporary address,
// so a stream left paused for long can lose its source.  Ask once more, at the same
// position; if that fails too, say so and offer the file.
function streamFailed() {
  if (!clip || !player.open) return;
  if (!retried) {
    retried = true;
    show(showing, reached, true);
    return;
  }
  big.pause();
  big.hidden = true;
  field(".player-error-link").href = clip.video[showing];
  failed.hidden = false;
}
source.addEventListener("error", streamFailed);
big.addEventListener("error", streamFailed);
big.addEventListener("timeupdate", () => {
  if (big.currentTime > 0) { reached = big.currentTime; retried = false; }
});

function openClip(data) {
  clip = data;
  retried = false;
  field("#player-title").textContent = clip.title;
  field(".player-about").textContent = clip.about;
  field(".player-length").textContent = length(clip.seconds) + ", looping";
  for (const a of player.querySelectorAll("[data-video]")) {
    const url = clip.video[a.dataset.video];
    a.parentElement.hidden = !url;
    if (url) a.href = url;
  }
  if (!player.open) player.showModal();
  updateAll();                          // the grid rests while the player is open
  show(isDark() ? "dark" : "light", 0, !reduceMotion.matches);
  history.replaceState(null, "", "#" + clip.name);
}

for (const b of player.querySelectorAll("[data-video-theme]")) {
  b.addEventListener("click", () => switchTo(b.dataset.videoTheme));
}
onTheme.push(() => switchTo(isDark() ? "dark" : "light"));

player.addEventListener("close", () => {
  clip = null;
  big.pause();
  source.removeAttribute("src");
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
