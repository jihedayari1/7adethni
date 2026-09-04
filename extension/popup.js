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

// MVP: translate is the hero feature and the default
let state = { feature:"translate", tone:"normal", apiUrl:DEFAULT_API, deviceId:null,
              lastReqId:null, lastGenerated:null, copiedThisReq:false };

const PLACEHOLDERS = {
  translate: "Ekteb bel français / english / العربية... w n7awlouhalek l derja 🇹🇳",
  reply:     "Colli el message elli jek, w n7adhroulek réponse bel tounsi 💬",
  rewrite:   "Ekteb el nass mte3ek heni, w n7assnouh w n5allouh a7la ✨",
};

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
  $("#input").placeholder = PLACEHOLDERS[state.feature] || PLACEHOLDERS.translate;
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

// ---- event logging (the data flywheel: copy / edit_copy / regen) ----
async function sendEvent(kind, payload){
  if(!state.lastReqId) return;
  try{ await fetch(api("/event"),{ method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ device_id:state.deviceId, usage_id:state.lastReqId, kind, payload:payload||null }) });
  }catch(_){ /* never block UX on logging */ }
}

// ---- generate ----
async function generate(){
  const text = $("#input").value.trim();
  if(!text){ setStatus("Ekteb 7aja luwel 🙂"); return; }
  // regenerating without copying = implicit rejection of the previous output
  if(state.lastReqId && !state.copiedThisReq) sendEvent("regen");
  const cta=$("#go"); cta.classList.add("is-loading"); cta.disabled=true;
  $("#output").classList.add("is-loading"); setStatus("9a3ed ye5dem… ⏳");
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
    state.lastGenerated = data.output||"";
    state.copiedThisReq = false;
    if(typeof data.remaining==="number") $("#quota").textContent = `${data.remaining}/${data.limit} 🎁`;
    setStatus("");
  }catch(e){ setStatus("Erreur: "+e.message+" — vérifie l'API fel ⚙️",true); }
  finally{ cta.classList.remove("is-loading"); cta.disabled=false; $("#output").classList.remove("is-loading"); }
}
$("#go").onclick = generate;
$("#regen").onclick = generate;

function escapeHtml(s){ return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
// gold-tint the Arabizi numerals (7 9 3 5 2) — textContent stays intact so edit-diffing still works
function paintNums(txt){ return escapeHtml(txt).replace(/[235789]/g, d => `<span class="num">${d}</span>`); }

function showOutput(txt){
  $("#output").innerHTML = paintNums(txt);
  $("#outBlock .out__tag").classList.add("hidden");           // drop the "Essai" badge
  $("#fbGood").classList.remove("is-on"); $("#fbFix").classList.remove("is-on");
}

// ---- copy: the strongest implicit signal. Edited-then-copied = a native correction. ----
$("#copy").onclick = async ()=>{
  const finalText = $("#output").textContent;
  await navigator.clipboard.writeText(finalText);
  state.copiedThisReq = true;
  if(state.lastGenerated !== null && finalText.trim() !== (state.lastGenerated||"").trim()){
    sendEvent("edit_copy", finalText);                        // implicit correction pair
    setStatus("T-copia — w 3aychek 3al tas7i7 ✍️📋");
  }else{
    sendEvent("copy");
    setStatus("T-copia 📋");
  }
};

// ---- explicit feedback ----
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
