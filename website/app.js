// voxanim website: plays each animation only while it is on screen, marks
// the current section in the step bar, and opens an animation in the player.
// The page works without it (posters, links and downloads); this only adds.
"use strict";

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const player = document.getElementById("player");

// ---------------------------------------------------------------- autoplay
// Videos load nothing until they first scroll into view, then play while
// visible.  With reduced motion they stay on their poster frame.

const visible = new Set();

function play(video) {
  if (reduceMotion.matches || player.open) return;
  if (!video.src) video.src = video.dataset.src;
  video.play().catch(() => {});   // autoplay refused: the poster stays
}

const onScreen = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) { visible.add(e.target); play(e.target); }
    else { visible.delete(e.target); e.target.pause(); }
  }
}, { threshold: 0.35 });
document.querySelectorAll("video[data-src]").forEach((v) => onScreen.observe(v));

reduceMotion.addEventListener("change", () => {
  visible.forEach((v) => (reduceMotion.matches ? v.pause() : play(v)));
});

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
  visible.forEach((v) => v.pause());
  big.src = clip.mp4;
  big.poster = clip.mp4.replace(/\.mp4$/, ".jpg");
  if (!player.open) player.showModal();
  if (!reduceMotion.matches) big.play().catch(() => {});
  history.replaceState(null, "", "#" + clip.name);
}

player.addEventListener("close", () => {
  big.pause();
  big.removeAttribute("src");
  big.load();
  history.replaceState(null, "", location.pathname + location.search);
  visible.forEach(play);
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
