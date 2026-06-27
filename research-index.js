const normalize=value=>value.normalize('NFKD').replace(/[\u0300-\u036f]/g,'').toLowerCase();

export function searchProjects(projects,query='',filters={}) {
 if(typeof query!=='string'||!Array.isArray(projects))throw new TypeError('invalid project search');
 const terms=[...new Set(normalize(query).split(/\s+/).filter(Boolean))];
 return projects.map(project=>{
  if(typeof project.id!=='string'||typeof project.title!=='string'||typeof project.summary!=='string'||!Array.isArray(project.tags))throw new TypeError('invalid project record');
  const title=normalize(project.title),summary=normalize(project.summary),tags=project.tags.map(normalize);
  const score=terms.reduce((sum,term)=>sum+(title.includes(term)?3:0)+(tags.some(tag=>tag.includes(term))?2:0)+(summary.includes(term)?1:0),0);
  const matches=terms.every(term=>title.includes(term)||summary.includes(term)||tags.some(tag=>tag.includes(term)));
  return {project,score,matches};
 }).filter(item=>item.matches&&(!filters.status||item.project.status===filters.status)&&(!filters.tag||item.project.tags.includes(filters.tag)))
 .sort((a,b)=>b.score-a.score||a.project.title.localeCompare(b.project.title))
 .map(item=>item.project);
}

export function facets(projects) {
 const counts=new Map();for(const project of projects)for(const tag of new Set(project.tags))counts.set(tag,(counts.get(tag)||0)+1);
 return [...counts].sort(([a],[b])=>a.localeCompare(b)).map(([tag,count])=>({tag,count}));
}

if(typeof document!=='undefined') {
 const grid=document.querySelector('main .grid');
 if(grid) {
  const cards=[...grid.querySelectorAll('.card')];
  const projects=cards.map((card,index)=>({id:String(index),title:card.querySelector('h2')?.textContent||'',summary:card.querySelector('h2 + p')?.textContent||'',tags:[card.querySelector('.tag')?.textContent||'Project'],status:card.querySelector('.status')?.textContent||''}));
  const controls=document.createElement('section');controls.setAttribute('aria-label','Find a research project');
  const label=document.createElement('label');label.textContent='Search projects ';const input=document.createElement('input');input.type='search';input.placeholder='Reasoning, evaluation, community…';label.append(input);
  const categoryLabel=document.createElement('label');categoryLabel.textContent=' Category ';const select=document.createElement('select');const all=document.createElement('option');all.value='';all.textContent='All categories';select.append(all);
  for(const facet of facets(projects)){const option=document.createElement('option');option.value=facet.tag;option.textContent=facet.tag+' ('+facet.count+')';select.append(option)}categoryLabel.append(select);
  const status=document.createElement('p');status.setAttribute('role','status');status.setAttribute('aria-live','polite');
  controls.append(label,categoryLabel,status);grid.before(controls);
  const update=()=>{const ids=new Set(searchProjects(projects,input.value,{tag:select.value}).map(project=>project.id));cards.forEach((card,index)=>card.hidden=!ids.has(String(index)));status.textContent=ids.size+' project'+(ids.size===1?'':'s')+' shown'};
  input.addEventListener('input',update);select.addEventListener('change',update);update();
 }
}
