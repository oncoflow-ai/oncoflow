import '@testing-library/jest-dom'
import { vi } from 'vitest'
import { demoUsers } from '@/store/authStore'

vi.mock('@/api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/client')>()
  return {
    ...actual,
    apiClient: {
      post: vi.fn(async (url, data) => {
        if (url === '/api/v1/auth/login') {
          let username = ''
          if (data instanceof URLSearchParams) {
            username = data.get('username') || ''
          }
          
          const normalized = username.trim().toLowerCase()
          let foundUser = demoUsers.find(u => 
            u.email.toLowerCase() === normalized || u.id.toLowerCase() === normalized
          )
          
          if (!foundUser && username === 'alex.reviewer@hospital.test') {
              foundUser = {
                  id: 'RAD-002',
                  name: 'Alex Reviewer',
                  email: 'alex.reviewer@hospital.test',
                  role: 'radiologist',
                  initials: 'AR'
              }
          }

          if (foundUser) {
            return {
              data: {
                access_token: 'mock-jwt-token',
                user: foundUser
              }
            }
          }
          throw new Error('Invalid user credentials')
        }
        return { data: {} }
      }),
      get: vi.fn().mockResolvedValue({ data: {} }),
      put: vi.fn().mockResolvedValue({ data: {} }),
      delete: vi.fn().mockResolvedValue({ data: {} }),
    }
  }
})
