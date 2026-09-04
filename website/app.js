/* 7adethni website v2 — theme, scroll-spy, reveals, animated transform demo, tone lab, inline mock */

// Set to your deployed gateway (backend) to use the REAL model. '' = offline demo dictionary.
const BACKEND_URL = ''; // e.g. 'https://you--7adethni-gateway.modal.run'

/* ---------- theme ---------- */
const root = document.documentElement,
      themeBtn = document.getElementById('themeBtn'),
      themeLbl = document.getElementById('themeLbl');
function setTheme(t){
  root.setAttribute('data-theme', t);
  themeLbl.textContent = t === 'dark' ? 'Light' : 'Dark';
  try{ localStorage.setItem('7adethni-theme', t); }catch(e){}
}
setTheme((()=>{ try{ return localStorage.getItem('7adethni-theme') || 'dark'; }catch(e){ return 'dark'; } })());
themeBtn.onclick = () => setTheme(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');

/* ---------- scroll reveals ---------- */
const revealer = new IntersectionObserver(es => es.forEach(e => {
  if(e.isIntersecting){ e.target.classList.add('in'); revealer.unobserve(e.target); }
}), { threshold: 0.12 });
document.querySelectorAll('.rv').forEach(el => revealer.observe(el));

/* ---------- scroll-spy nav ---------- */
const tabLinks = [...document.querySelectorAll('.tabs a')];
const spy = new IntersectionObserver(es => es.forEach(e => {
  if(e.isIntersecting){
    tabLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
  }
}), { rootMargin: '-40% 0px -55% 0px' });
['overview','onboarding','popup','sidepanel','inline','tones'].forEach(id => {
  const el = document.getElementById(id); if(el) spy.observe(el);
});

/* ---------- offline demo dictionary (fallback when BACKEND_URL is empty) ---------- */
const PHRASES = [
  ['comment ça va mon ami','chnowa el 7keya ya sa7bi'],['comment ça va','chnowa el 7keya'],
  ['comment ca va','chnowa el 7keya'],['how are you','chnia a7welek'],
  ['i love you','n7ebbek barcha'],['good morning','sba7 el 5ir'],['thank you','3aychek barcha'],
  ['my friend','ya sa7bi'],['god willing','inchallah'],['i am late','rani m3attel'],
  ["i'm late",'rani m3attel'],['see you tomorrow','nchoufek 8odwa'],['see you','nchoufek'],
  ["i can't come tonight",'ma nnajamch nji el lila'],['sans mots','bla klem'],
  ['tellement magnifique','7aja ma tetsawarch'],
];
const DICT = {bonjour:'3aslema',salut:'3aslema',hello:'3aslema',hi:'3aslema',hey:'3aslema',coucou:'3aslema',
  merci:'3aychek',thanks:'3aychek',thank:'3aychek',oui:'ey',yes:'ey',ok:'mre8el',non:"le'",no:"le'",
  friend:'sa7bi',ami:'sa7bi',copain:'sa7bi',brother:'5ouya',what:'chnowa',quoi:'chnowa',
  beautiful:'7elw',belle:'7elwa',beau:'7elw',jolie:'7elwa',magnifique:'7elw barcha',
  enough:'yezzi',stop:'yezzi',very:'barcha',much:'barcha',beaucoup:'barcha',trop:'barcha',
  love:'n7eb',aime:'n7eb',good:'behi',cool:'behi',great:'behi',nice:'behi',bien:'behi',
  coffee:'9ahwa',cafe:'9ahwa','café':'9ahwa',sorry:'sma7ni','désolé':'sma7ni',desole:'sma7ni',
  pardon:'sma7ni',late:'m3attel',retard:'te2khir',story:'7keya',histoire:'7keya',sea:'b7ar',mer:'b7ar',
  sun:'chams',soleil:'chams',please:'3aychek',now:'tawa',maintenant:'tawa',tomorrow:'8odwa',demain:'8odwa',
  today:'lyoum',tonight:'el lila',work:'5edma',travail:'5edma',house:'dar',maison:'dar',
  sunset:'8roub',coucher:'8roub',soleil2:'chams',unreal:'5ural'};

const TONE_WRAP = {
  normal: s => s,
  funny:  s => s + ' 😂 wallah 8alba',
  formal: s => s.charAt(0).toUpperCase() + s.slice(1) + '.',
};

function mockTranslate(input){
  if(!input || !input.trim()) return '';
  let s = ' ' + input.toLowerCase().trim() + ' ';
  for(const [a,b] of PHRASES) s = s.split(' ' + a + ' ').join(' ' + b + ' ');
  s = s.replace(/[.,!?؟]/g, '').trim();
  return s.split(/\s+/).map(w => DICT[w] || w).join(' ');
}
function paintNums(text){
  return text.replace(/[235789]/g, d => `<span class="num">${d}</span>`);
}

/* ---------- backend call (real model when deployed) ---------- */
function deviceId(){
  try{
    let id = localStorage.getItem('7adethni-device');
    if(!id){ id = 'web-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
      localStorage.setItem('7adethni-device', id); }
    return id;
  }catch(e){ return 'web-anon'; }
}
async function callModel(text, tone){
  if(!BACKEND_URL) return null;
  try{
    const r = await fetch(BACKEND_URL.replace(/\/$/,'') + '/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ feature:'translate', tone, text, device_id: deviceId() })
    });
    if(!r.ok) return null;
    const j = await r.json();
    return j.output || null;
  }catch(e){ return null; }
}

/* ---------- hero demo: typing animation ---------- */
const srcInput = document.getElementById('srcInput'),
      outText  = document.getElementById('outText'),
      goBtn    = document.getElementById('goBtn'),
      replay   = document.getElementById('replay'),
      toneRow  = document.getElementById('toneRow');
let activeTone = 'normal', typeTimer = null;

function typeOut(text){
  clearInterval(typeTimer);
  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(reduced){ outText.innerHTML = paintNums(text); return; }
  let i = 0;
  typeTimer = setInterval(() => {
    i++;
    outText.innerHTML = paintNums(text.slice(0, i)) + (i < text.length ? '<span class="caret"></span>' : '');
    if(i >= text.length) clearInterval(typeTimer);
  }, 26);
}

async function transform(){
  const text = srcInput.value.trim();
  if(!text){ outText.textContent = ''; return; }
  goBtn.disabled = true;
  let out = await callModel(text, activeTone);            // real model if deployed
  if(out == null) out = (TONE_WRAP[activeTone] || (s=>s))(mockTranslate(text));
  typeOut(out);
  goBtn.disabled = false;
}
goBtn.onclick = transform;
replay.onclick = () => { srcInput.value = 'Comment ça va mon ami ?'; activeTone = 'normal';
  [...toneRow.children].forEach((b,i)=>b.classList.toggle('active', i===0)); transform(); };
srcInput.addEventListener('keydown', e => { if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); transform(); } });
toneRow.addEventListener('click', e => {
  const b = e.target.closest('.tone'); if(!b) return;
  [...toneRow.children].forEach(x => x.classList.remove('active')); b.classList.add('active');
  activeTone = b.dataset.tone; transform();
});

/* ---------- tone lab (interactive) ---------- */
const TONELAB = [
  ['3adi',      "sma7ni, ma nnajamch nji el lila 🙏"],
  ['Morfeh 😄', "ya sa7bi el lila mch lila, el canapé rb7ni 😂 nchoufek 8odwa inchallah"],
  ['Rasmi',     "Sma7ni barcha, 3andi empêchement el lila. Net9ablou nhar a5er inchallah."],
  ['7anin 🤍',  "sma7ni 3zizi, el lila ma nnajamch... ama el 9alb m3ak 🤍"],
];
const tonelabTabs = document.getElementById('tonelabTabs'),
      tonelabOut  = document.getElementById('tonelabOut');
TONELAB.forEach(([label], i) => {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'tone' + (i === 0 ? ' active' : ''); b.textContent = label;
  b.onclick = () => {
    [...tonelabTabs.children].forEach(x => x.classList.remove('active')); b.classList.add('active');
    tonelabOut.style.opacity = 0;
    setTimeout(() => { tonelabOut.innerHTML = paintNums(TONELAB[i][1]);
      tonelabOut.style.transition = 'opacity .3s'; tonelabOut.style.opacity = 1; }, 120);
  };
  tonelabTabs.appendChild(b);
});
tonelabOut.innerHTML = paintNums(TONELAB[0][1]);

/* ---------- inline mock ---------- */
const inlineGo = document.getElementById('inlineGo'),
      inlineText = document.getElementById('inlineText');
if(inlineGo) inlineGo.onclick = () => {
  const sel = inlineText.querySelector('.sel');
  if(sel){ sel.innerHTML = paintNums('7aja ma tetsawarch, bla klem'); sel.style.background = 'oklch(0.86 0.09 88 / .25)'; }
  inlineGo.textContent = 'Behi ✓'; inlineGo.disabled = true;
};

/* ---------- install button (beta note until Web Store) ---------- */
document.getElementById('installBtn').onclick = (e) => {
  e.preventDefault();
  alert("Beta 🔥 el extension jeya 9rib lel Chrome Web Store.\nEnti mel awelin — jarrab el demo w khallik 9rib!");
};

/* autoplay hero once */
window.addEventListener('load', () => setTimeout(transform, 600));
