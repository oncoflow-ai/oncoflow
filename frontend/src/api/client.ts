import axios from 'axios'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
})

// JWT interceptor stub — no-op in mock phase.
// When backend is ready, uncomment:
// apiClient.interceptors.request.use(cfg => {
//   const token = sessionStorage.getItem('oncoflow_token')
//   if (token) cfg.headers.Authorization = `Bearer ${token}`
//   return cfg
// })

export const delay = (ms: number): Promise<void> => new Promise(res => setTimeout(res, ms))
