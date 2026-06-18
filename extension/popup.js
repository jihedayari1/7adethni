const $ = (s) => document.querySelector(s);
const DEFAULT_API = "http://localhost:9000";

// ---- storage: chrome.storage in the extension, localStorage when previewing in a browser ----
const hasChrome = typeof chrome !== "undefined" && chrome.storage && chrome.storage.sync;
const store = {
  async get(keys){ if(hasChrome) return new Promise(r=>chrome.storage.sync.get(keys,r));
    const o={}; for(const k in keys){ const v=localStorage.getItem(k); o[k]=v===null?keys[k]:v; } return o; },
  async set(obj){ if(hasChrome) return new Promise(r=>chrome.storage.sync.set(obj,r));
    for(const k in obj) localStorage.setItem(k,obj[k]); }
};

let state = { feature:"post", tone:"normal", apiUrl:DEFAULT_API, deviceId:null, lastReqId:null };

async function init(){
  const s = await store.get({ apiUrl:DEFAULT_API, deviceId:"" });
  state.apiUrl = s.apiUrl || DEFAULT_API;
  state.deviceId = s.deviceId || (crypto.randomUUID ? crypto.randomUUID() : "dev-"+Date.now()+Math.random());
  if(!s.deviceId) await store.set({ deviceId: state.deviceId });
  $("#apiUrl").value = state.apiUrl;
  refreshQuota();
}

// ---- feature & tone selection ----
$("#features").addEventListener("click", e=>{
  const b=e.target.closest(".pill"); if(!b) return;
  $("#features .is-active")?.classList.remove("is-active"); b.classList.add("is-active");
  state.feature=b.dataset.feat;
});
$("#tones").addEventListener("click", e=>{
  const b=e.target.closest(".tone"); if(!b) return;
  $("#tones .is-active")?.classList.remove("is-active"); b.classList.add("is-active");
  state.tone=b.dataset.tone;
});

// ---- settings ----
$("#settingsBtn").onclick = ()=> $("#settings").classList.toggle("hidden");
$("#saveSettings").onclick = async ()=>{
  state.apiUrl = ($("#apiUrl").value.trim()||DEFAULT_API);
  await store.set({ apiUrl: state.apiUrl });
  $("#settings").classList.add("hidden"); setStatus("Réglages saved ✅"); refreshQuota();
};

function setStatus(msg, err=false){ const el=$("#status"); el.textContent=msg||""; el.classList.toggle("err",err); }
function api(path){ return `${state.apiUrl.replace(/\/$/,"")}${path}`; }

async function refreshQuota(){
  try{
    const r = await fetch(api(`/me?device_id=${encodeURIComponent(state.deviceId)}`));
    if(r.ok){ const q=await r.json(); $("#quota").textContent = `${q.remaining}/${q.limit} 🎁`; }
  }catch(_){ /* offline preview: leave default */ }
}

// ---- generate ----
async function generate(){
  const text = $("#input").value.trim();
  if(!text){ setStatus("Ekteb 7aja luwel 🙂"); return; }
  const cta=$("#go"); cta.classList.add("is-loading"); cta.disabled=true; setStatus("9a3ed ye5dem… ⏳");
  try{
    const r = await fetch(api("/generate"),{
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({ device_id:state.deviceId, feature:state.feature, tone:state.tone, text })
    });
    if(r.status===429){ const q=await r.json().catch(()=>({}));
      setStatus((q.detail&&q.detail.message)||"5lset el génération mejjeniya el yom 🎁",true); return; }
    if(!r.ok) throw new Error("API "+r.status);
    const data = await r.json();
    showOutput(data.output||"(walou)");
    state.lastReqId = data.request_id;
    if(typeof data.remaining==="number") $("#quota").textContent = `${data.remaining}/${data.limit} 🎁`;
    setStatus("");
  }catch(e){ setStatus("Erreur: "+e.message+" — vérifie l'API fel ⚙️",true); }
  finally{ cta.classList.remove("is-loading"); cta.disabled=false; }
}
$("#go").onclick = generate;
$("#regen").onclick = generate;

function showOutput(txt){
  $("#output").textContent = txt;
  $("#outBlock .out__tag").classList.add("hidden");           // drop the "Essai" badge
  $("#fbGood").classList.remove("is-on"); $("#fbFix").classList.remove("is-on");
}

// ---- copy + feedback (the data flywheel) ----
$("#copy").onclick = async ()=>{ await navigator.clipboard.writeText($("#output").textContent); setStatus("T-copia 📋"); };
$("#fbGood").onclick = ()=>{ sendFeedback("good"); $("#fbGood").classList.add("is-on"); setStatus("3aychek 🤍"); };
$("#fbFix").onclick = ()=>{
  const corrected = prompt("Sa77a7 el nass (kifech el sa7i7?):", $("#output").textContent);
  if(corrected===null) return;
  sendFeedback("bad", corrected); $("#fbFix").classList.add("is-on"); setStatus("Barakallahou fik, t-7afdhet ✍️");
};
async function sendFeedback(rating, corrected){
  if(!state.lastReqId) return;
  try{ await fetch(api("/feedback"),{ method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ request_id:state.lastReqId, rating, corrected:corrected||null }) }); }catch(_){}
}

init();
