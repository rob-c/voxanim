// voxanim glossary: the theme button, and filtering the table by words and topic.
// The page works without this script: the whole table is there.
"use strict";

const root = document.documentElement;
const themeButton = document.getElementById("theme");
const systemDark = matchMedia("(prefers-color-scheme: dark)");
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

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
  const next = isDark() ? "light" : "dark";
  store.set("theme", next);
  applyTheme(next);
});
systemDark.addEventListener("change", () => applyTheme(root.dataset.theme || null));

const search = document.getElementById("gloss-search");
const topic = document.getElementById("gloss-topic");
const shown = document.getElementById("gloss-shown");
const rows = [...document.querySelectorAll("#glossary tbody tr")];

function filter() {
  const words = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const want = topic.value;
  let n = 0;
  for (const row of rows) {
    const text = row.dataset.text;
    const ok = (!want || row.dataset.topic === want) && words.every((w) => text.includes(w));
    row.hidden = !ok;
    n += ok;
  }
  shown.textContent = n;
}

search.addEventListener("input", filter);
topic.addEventListener("change", filter);
const q = new URLSearchParams(location.search).get("q");
if (q) { search.value = q; filter(); }
