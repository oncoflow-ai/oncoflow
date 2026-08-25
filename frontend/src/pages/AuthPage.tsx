import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import { ROLE_HOME } from '@/lib/routes'

type Mode = 'signin' | 'register'

export default function AuthPage() {
  const [mode, setMode] = useState<Mode>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const login = useAuthStore(s => s.login)
  const navigate = useNavigate()

  async function handleSignIn(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const user = await login(email, password)
      navigate(ROLE_HOME[user.role])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign in failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-bg flex items-stretch">
      <div className="hidden lg:flex flex-col justify-between flex-1 bg-surface border-r border-border p-14 relative overflow-hidden">
        <div className="absolute -top-10 -left-10 w-72 h-72 rounded-full bg-teal/10 blur-[80px] pointer-events-none" />
        <div>
          <div className="text-[28px] font-sans font-bold text-text1 tracking-tight flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-teal shadow-[0_0_8px_#0DC5A0]" />
            OncoFlow
          </div>
          <div className="text-[11px] font-mono text-text3 tracking-widest uppercase mt-1.5 ml-4">
            Longitudinal Tumor Intelligence
          </div>
        </div>

        <div>
          <h1 className="text-[44px] font-sans font-bold text-text1 leading-[1.15] mb-5">
            Precision<br />tracking for<br />
            <span className="italic text-teal">every scan.</span>
          </h1>
          <p className="text-[14px] text-text2 leading-relaxed max-w-sm">
            Automated tumor segmentation, volumetric comparison, and AI-generated clinical narratives — built for oncologists, radiologists, and patient-facing review.
          </p>
        </div>

        <div className="flex gap-8">
          {[
            { num: '98.4%', label: 'Seg. accuracy' },
            { num: '312', label: 'Reports generated' },
            { num: '3', label: 'AI models (ensemble)' },
          ].map(stat => (
            <div key={stat.label}>
              <div className="font-mono text-[26px] font-bold text-text1">{stat.num}</div>
              <div className="text-[11px] font-mono text-text3 uppercase tracking-widest mt-0.5">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="w-full lg:w-[380px] flex flex-col justify-center px-10 py-16 bg-bg">
        {mode === 'signin' ? (
          <form onSubmit={handleSignIn} className="space-y-4">
            <div className="text-[12px] font-mono font-bold tracking-widest uppercase text-text2 mb-8">
              Clinical Sign In
            </div>

            <div>
              <label className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="dr.cohen@ichilov.gov.il"
                autoComplete="email"
                className="w-full bg-surface border border-border2 text-text1 px-3.5 py-[10px] text-[14px] font-sans placeholder-text3 focus:outline-none focus:border-teal transition-colors"
              />
            </div>
            <div>
              <label className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
                autoComplete="current-password"
                className="w-full bg-surface border border-border2 text-text1 px-3.5 py-[10px] text-[14px] font-sans placeholder-text3 focus:outline-none focus:border-teal transition-colors"
              />
            </div>
            {error && (
              <p className="text-danger text-[12px] font-mono">{error}</p>
            )}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-teal text-black font-mono font-bold text-[13px] tracking-widest uppercase py-3 mt-2 hover:bg-teal/90 transition-colors disabled:opacity-50"
            >
              {loading ? 'Signing in…' : 'Continue →'}
            </button>
            <div className="border border-border2 bg-surface p-3.5 space-y-2 text-[11px] font-mono">
              <div className="text-text2 font-bold tracking-wider uppercase mb-1">
                Quick Demo Logins:
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  onClick={() => { setEmail('admin@oncoflow.local'); setPassword('admin123'); }}
                  className="px-2 py-1.5 bg-bg border border-border hover:border-teal text-left transition-colors"
                >
                  <div className="text-teal font-bold">Admin</div>
                  <div className="text-text3 text-[10px] truncate">admin@oncoflow.local</div>
                </button>

                <button
                  type="button"
                  onClick={() => { setEmail('radiology@oncoflow.local'); setPassword('password'); }}
                  className="px-2 py-1.5 bg-bg border border-border hover:border-teal text-left transition-colors"
                >
                  <div className="text-teal font-bold">Radiologist</div>
                  <div className="text-text3 text-[10px] truncate">radiology@oncoflow.local</div>
                </button>

                <button
                  type="button"
                  onClick={() => { setEmail('dr.cohen@ichilov.gov.il'); setPassword('password'); }}
                  className="px-2 py-1.5 bg-bg border border-border hover:border-teal text-left col-span-2 transition-colors"
                >
                  <div className="text-teal font-bold">Doctor</div>
                  <div className="text-text3 text-[10px] truncate">dr.cohen@ichilov.gov.il</div>
                </button>

                <button
                  type="button"
                  onClick={() => { setEmail('david.levi@example.test'); setPassword('patient123'); }}
                  className="px-2 py-1.5 bg-bg border border-border hover:border-teal text-left col-span-2 transition-colors"
                >
                  <div className="text-teal font-bold">Patient (Portal) — David Levi</div>
                  <div className="text-text3 text-[10px] truncate">david.levi@example.test (or ID PAT-1031)</div>
                </button>
              </div>
            </div>
            <hr className="border-border my-6" />
            <p className="text-[12px] text-text2 text-center font-sans">
              Need access?{' '}
              <button type="button" onClick={() => setMode('register')} className="text-teal font-semibold">
                Request access from admin →
              </button>
            </p>
          </form>
        ) : (
          <div>
            <div className="text-[12px] font-mono font-bold tracking-widest uppercase text-text2 mb-8">
              Request Access
            </div>
            <p className="text-[13px] text-text2 mb-6 leading-relaxed">
              Submit your details and a medical administrator will grant access within 24 hours.
            </p>
            {['Full Name', 'Hospital Email', 'Hospital ID', 'Role'].map(field => (
              <div key={field} className="mb-4">
                <label className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                  {field}
                </label>
                <input
                  placeholder=""
                  className="w-full bg-surface border border-border2 text-text1 px-3.5 py-[10px] text-[14px] font-sans focus:outline-none focus:border-teal transition-colors"
                />
              </div>
            ))}
            <button type="button" className="w-full bg-surface border border-teal text-teal font-mono font-bold text-[13px] tracking-widest uppercase py-3 hover:bg-teal/5 transition-colors">
              Submit Request
            </button>
            <hr className="border-border my-6" />
            <p className="text-[12px] text-text2 text-center font-sans">
              Already have access?{' '}
              <button type="button" onClick={() => setMode('signin')} className="text-teal font-semibold">
                Sign in →
              </button>
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
