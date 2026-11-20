const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
const token = () => localStorage.getItem('sb_token');
async function request(path, options={}) {
  const headers={'Content-Type':'application/json', ...(options.headers||{})};
  if(token()) headers.Authorization=`Bearer ${token()}`;
  const res=await fetch(`${API}${path}`,{...options,headers});
  const data=await res.json().catch(()=>({detail:'Unexpected server response'}));
  if(!res.ok) throw new Error(data.detail||'Request failed');
  return data;
}
export const api={
 register:(body)=>request('/auth/register',{method:'POST',body:JSON.stringify(body)}),
 login:(body)=>request('/auth/login',{method:'POST',body:JSON.stringify(body)}),
 me:()=>request('/auth/me'),
 profile:()=>request('/profile'), saveProfile:(body)=>request('/profile',{method:'PUT',body:JSON.stringify(body)}),
 dashboard:()=>request('/dashboard'), scholarships:(params='')=>request(`/scholarships${params?`?${params}`:''}`), scholarship:(id)=>request(`/scholarships/${id}`),
 recommend:(profile)=>request('/recommend',{method:'POST',body:JSON.stringify({profile})}),
 rag:(body)=>request('/rag/search',{method:'POST',body:JSON.stringify(body)}),
 explain:(body)=>request('/explain',{method:'POST',body:JSON.stringify(body)}),
 workflow:(body)=>request('/workflow',{method:'POST',body:JSON.stringify(body)}), saved:()=>request('/saved'), save:(id)=>request('/saved',{method:'POST',body:JSON.stringify({scholarship_id:id})}), unsave:(id)=>request(`/saved/${id}`,{method:'DELETE'})
};
