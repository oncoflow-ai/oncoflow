import { Navigate } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import { ROLE_HOME } from '@/lib/routes'

/** Legacy route: forwards to the home screen for the signed-in role. */
export default function DashboardPage() {
  const user = useAuthStore(s => s.user)
  if (!user) return <Navigate to="/auth" replace />
  return <Navigate to={ROLE_HOME[user.role]} replace />
}
