import { createBrowserRouter, Navigate, Outlet } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import AuthPage from '@/pages/AuthPage'
import DashboardPage from '@/pages/DashboardPage'
import PatientDetailPage from '@/pages/PatientDetailPage'

function ProtectedRoute() {
  const physician = useAuthStore(s => s.physician)
  return physician !== null ? <Outlet /> : <Navigate to="/auth" replace />
}

function AuthGuard() {
  const physician = useAuthStore(s => s.physician)
  return physician !== null ? <Navigate to="/dashboard" replace /> : <AuthPage />
}

export const router = createBrowserRouter([
  { path: '/', element: <Navigate to="/dashboard" replace /> },
  { path: '/auth', element: <AuthGuard /> },
  {
    element: <ProtectedRoute />,
    children: [
      { path: '/dashboard', element: <DashboardPage /> },
      { path: '/patients/:id', element: <PatientDetailPage /> },
    ],
  },
])
