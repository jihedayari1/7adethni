const $ = (id) => document.getElementById(id);
const DEFAULTS = { apiUrl: "http://localhost:8000", apiKey: "" };

// ---- settings (stored in chrome.storage) ----
function loadSettings() {
  return new Promise((res) => chrome.storage.sync.get(DEFAULTS, res));
}
$("settingsBtn").onclick = () => $("settings").classList.toggle("hidden");
$("saveSettings").onclick = async () => {
  await chrome.storage.sync.set({ apiUrl: $("apiUrl").value.trim(), apiKey: $("apiKey").value.trim() });
  $("settings").classList.add("hidden");
  setStatus("Settings saved ✅");
};

function setStatus(msg) { $("status").textContent = msg || ""; }

// ---- generate ----
$("go").onclick = async () => {
  const text = $("input").value.trim();
  if (!text) { setStatus("Ekteb 7aja luwel."); return; }
  const { apiUrl, apiKey } = await loadSettings();
  $("go").disabled = true; setStatus("9a3ed ye5dem... ⏳");
  $("output").classList.add("hidden"); $("copy").classList.add("hidden");
  try {
    const r = await fetch(`${apiUrl.replace(/\/$/, "")}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(apiKey ? { "x-api-key": apiKey } : {}) },
      body: JSON.stringify({ feature: $("feature").value, tone: $("tone").value, text }),
    });
    if (!r.ok) throw new Error(`API ${r.status}`);
    const data = await r.json();
    $("output").textContent = data.output || "(walou)";
    $("output").classList.remove("hidden"); $("copy").classList.remove("hidden");
    setStatus("");
  } catch (e) {
    setStatus("Erreur: " + e.message + " — vérifie l'API URL fel ⚙️.");
  } finally {
    $("go").disabled = false;
  }
};

$("copy").onclick = async () => {
  await navigator.clipboard.writeText($("output").textContent);
  setStatus("T-copia 📋");
};

// preload saved API URL into the settings inputs
loadSettings().then((s) => { $("apiUrl").value = s.apiUrl; $("apiKey").value = s.apiKey; });
