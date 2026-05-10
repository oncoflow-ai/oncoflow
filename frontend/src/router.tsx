import { createBrowserRouter, Navigate, Outlet } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import type { UserRole } from '@/types'
import { ROLE_HOME } from '@/lib/routes'
import AuthPage from '@/pages/AuthPage'
import DashboardPage from '@/pages/DashboardPage'
import PatientDetailPage from '@/pages/PatientDetailPage'
import DoctorDashboardPage from '@/pages/DoctorDashboardPage'
import DoctorPatientDashboardPage from '@/pages/DoctorPatientDashboardPage'
import RadiologistWorkspacePage from '@/pages/RadiologistWorkspacePage'
import PatientPortalPage from '@/pages/PatientPortalPage'
import AdminUsersPage from '@/pages/AdminUsersPage'
import RadiologyReviewPage from '@/pages/RadiologyReviewPage'

function ProtectedRoute() {
  const user = useAuthStore(s => s.user)
  return user !== null ? <Outlet /> : <Navigate to="/auth" replace />
}

function AuthGuard() {
  const user = useAuthStore(s => s.user)
  if (user !== null) {
    return <Navigate to={ROLE_HOME[user.role]} replace />
  }
  return <AuthPage />
}

function RoleRoute({ role, roles }: { role?: UserRole; roles?: UserRole[] }) {
  const user = useAuthStore(s => s.user)
  if (user === null) return <Navigate to="/auth" replace />
  const allowedRoles = roles ?? (role ? [role] : [])
  if (!allowedRoles.includes(user.role)) return <Navigate to={ROLE_HOME[user.role]} replace />
  return <Outlet />
}

function RootRedirect() {
  const user = useAuthStore(s => s.user)
  if (user === null) return <Navigate to="/auth" replace />
  return <Navigate to={ROLE_HOME[user.role]} replace />
}

export const router = createBrowserRouter([
  { path: '/', element: <RootRedirect /> },
  { path: '/auth', element: <AuthGuard /> },
  {
    element: <ProtectedRoute />,
    children: [
      { path: '/dashboard', element: <DashboardPage /> },
      { path: '/patients/:id', element: <PatientDetailPage /> },
      {
        element: <RoleRoute role="admin" />,
        children: [{ path: '/admin/users', element: <AdminUsersPage /> }],
      },
      {
        element: <RoleRoute roles={['doctor', 'clinician']} />,
        children: [
          { path: '/doctor', element: <DoctorDashboardPage /> },
          { path: '/doctor/patients/:id', element: <DoctorPatientDashboardPage /> },
        ],
      },
      {
        element: <RoleRoute role="radiologist" />,
        children: [
          { path: '/radiologist', element: <RadiologistWorkspacePage /> },
          { path: '/review', element: <RadiologyReviewPage /> },
        ],
      },
      {
        element: <RoleRoute role="patient" />,
        children: [
          { path: '/patient', element: <PatientPortalPage /> },
          { path: '/portal', element: <PatientPortalPage /> },
        ],
      },
    ],
  },
])
