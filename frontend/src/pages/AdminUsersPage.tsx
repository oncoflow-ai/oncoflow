import { useState } from 'react'
import { ShieldCheck, UserPlus } from 'lucide-react'
import TopNav from '@/components/layout/TopNav'
import { useAuthStore } from '@/store/authStore'
import { roleLabels } from '@/lib/roles'
import type { UserRole } from '@/types'
import { mockPatients } from '@/data/mockData'
import { getPatientOwnerId } from '@/lib/patientAccess'

const roleOptions: UserRole[] = ['doctor', 'radiologist', 'clinician', 'patient', 'admin']

export default function AdminUsersPage() {
  const users = useAuthStore(s => s.users)
  const addUser = useAuthStore(s => s.addUser)
  const assignPatient = useAuthStore(s => s.assignPatient)
  const patientAssignments = useAuthStore(s => s.patientAssignments)
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [role, setRole] = useState<UserRole>('doctor')
  const [patientRecordId, setPatientRecordId] = useState(mockPatients[0]?.id ?? '')
  const [error, setError] = useState('')
  const [created, setCreated] = useState('')
  const clinicalUsers = users.filter(user => user.role === 'doctor' || user.role === 'clinician')

  async function handleCreateUser(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setCreated('')

    try {
      const user = await addUser({
        name,
        email,
        password,
        role,
        patientRecordId: role === 'patient' ? patientRecordId : undefined,
      })
      setName('')
      setEmail('')
      setPassword('')
      setRole('doctor')
      setPatientRecordId(mockPatients[0]?.id ?? '')
      setCreated(`${user.name} can now sign in as ${roleLabels[user.role]}.`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create user')
    }
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <TopNav />

      <main className="flex-1 px-5 py-6">
        <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
          <form onSubmit={handleCreateUser} className="bg-surface border border-border p-5 h-fit">
            <div className="flex items-center gap-2 text-text1">
              <UserPlus size={17} />
              <h1 className="text-[14px] font-mono font-bold uppercase tracking-widest">
                Add User
              </h1>
            </div>

            <div className="mt-5 space-y-4">
              <label className="block">
                <span className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                  Full Name
                </span>
                <input
                  value={name}
                  onChange={e => setName(e.target.value)}
                  className="w-full bg-bg border border-border2 text-text1 px-3.5 py-[10px] text-[14px] focus:outline-none focus:border-teal transition-colors"
                />
              </label>

              <label className="block">
                <span className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                  Email
                </span>
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  className="w-full bg-bg border border-border2 text-text1 px-3.5 py-[10px] text-[14px] focus:outline-none focus:border-teal transition-colors"
                />
              </label>

              <label className="block">
                <span className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                  Temporary Password
                </span>
                <input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="w-full bg-bg border border-border2 text-text1 px-3.5 py-[10px] text-[14px] focus:outline-none focus:border-teal transition-colors"
                />
              </label>

              <label className="block">
                <span className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                  Role
                </span>
                <select
                  value={role}
                  onChange={e => setRole(e.target.value as UserRole)}
                  className="w-full bg-bg border border-border2 text-text1 px-3.5 py-[10px] text-[14px] focus:outline-none focus:border-teal transition-colors"
                >
                  {roleOptions.map(option => (
                    <option key={option} value={option}>
                      {roleLabels[option]}
                    </option>
                  ))}
                </select>
              </label>

              {role === 'patient' && (
                <label className="block">
                  <span className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                    Linked Patient Record
                  </span>
                  <select
                    value={patientRecordId}
                    onChange={e => setPatientRecordId(e.target.value)}
                    className="w-full bg-bg border border-border2 text-text1 px-3.5 py-[10px] text-[14px] focus:outline-none focus:border-teal transition-colors"
                  >
                    {mockPatients.map(patient => (
                      <option key={patient.id} value={patient.id}>
                        {patient.name} · {patient.id}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>

            {error && <p className="text-danger text-[12px] font-mono mt-4">{error}</p>}
            {created && <p className="text-teal text-[12px] font-mono mt-4">{created}</p>}

            <button
              type="submit"
              className="w-full bg-teal text-black font-mono font-bold text-[13px] tracking-widest uppercase py-3 mt-5 hover:bg-teal/90 transition-colors"
            >
              Create User
            </button>
          </form>

          <section className="bg-surface border border-border">
            <div className="px-5 py-3.5 border-b border-border flex items-center justify-between">
              <span className="text-[11px] font-mono text-text3 uppercase tracking-widest">
                {users.length} authorized user{users.length !== 1 ? 's' : ''}
              </span>
              <span className="text-[11px] font-mono text-text3">Role-based routing active</span>
            </div>

            <div className="divide-y divide-border">
              {users.map(user => (
                <div key={user.email} className="px-5 py-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="w-9 h-9 bg-surface2 border border-border2 flex items-center justify-center font-mono text-[11px] text-text2 shrink-0">
                      {user.initials}
                    </div>
                    <div className="min-w-0">
                      <div className="text-text1 text-[14px] font-semibold truncate">{user.name}</div>
                      <div className="text-text3 text-[12px] font-mono truncate">
                        {user.email}
                        {user.patientRecordId ? ` · linked to ${user.patientRecordId}` : ''}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 md:justify-end">
                    <span className="inline-flex items-center gap-1.5 border border-border2 bg-bg px-2.5 py-1 text-[11px] font-mono uppercase tracking-widest text-text2">
                      <ShieldCheck size={12} />
                      {roleLabels[user.role]}
                    </span>
                    <span className="text-[11px] font-mono text-text3">{user.id}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="bg-surface border border-border xl:col-span-2">
            <div className="px-5 py-3.5 border-b border-border flex items-center justify-between">
              <span className="text-[11px] font-mono text-text3 uppercase tracking-widest">
                Patient Ownership
              </span>
              <span className="text-[11px] font-mono text-text3">
                Controls which user can see each patient
              </span>
            </div>

            <div className="divide-y divide-border">
              {mockPatients.map(patient => {
                const ownerId = getPatientOwnerId(patient, patientAssignments)
                return (
                  <div key={patient.id} className="px-5 py-4 grid gap-3 md:grid-cols-[1fr_280px] md:items-center">
                    <div className="min-w-0">
                      <div className="text-text1 text-[14px] font-semibold truncate">
                        {patient.name}
                      </div>
                      <div className="text-text3 text-[12px] font-mono truncate">
                        {patient.id} · {patient.diagnosis} · {patient.diagnosisLocation}
                      </div>
                    </div>
                    <select
                      value={ownerId}
                      onChange={e => assignPatient(patient.id, e.target.value)}
                      className="w-full bg-bg border border-border2 text-text1 px-3.5 py-[10px] text-[13px] focus:outline-none focus:border-teal transition-colors"
                      aria-label={`Owner for ${patient.name}`}
                    >
                      {clinicalUsers.map(user => (
                        <option key={user.id} value={user.id}>
                          {user.name} · {roleLabels[user.role]}
                        </option>
                      ))}
                    </select>
                  </div>
                )
              })}
            </div>
          </section>
        </div>
      </main>
    </div>
  )
}
