import { Navigate, useParams } from 'react-router-dom'

/** Legacy route: canonical doctor patient URL is `/doctor/patients/:id`. */
export default function PatientDetailPage() {
  const { id } = useParams<{ id: string }>()
  if (!id) return <Navigate to="/doctor" replace />
  return <Navigate to={`/doctor/patients/${id}`} replace />
}
