import { LogOut } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'

interface TopNavProps {
  searchValue?: string
  onSearchChange?: (value: string) => void
  showSearch?: boolean
  cta?: React.ReactNode
}

export default function TopNav({ searchValue, onSearchChange, showSearch = false, cta }: TopNavProps) {
  const { physician, logout } = useAuthStore()
  const navigate = useNavigate()

  function handleLogout() {
    logout()
    navigate('/auth')
  }

  return (
    <header className="h-[52px] bg-bg border-b border-border flex items-center gap-5 px-5 shrink-0">
      <span className="font-sans font-semibold text-[17px] text-text1 tracking-tight whitespace-nowrap flex items-center gap-2">
        <span className="w-[7px] h-[7px] rounded-full bg-teal shadow-[0_0_8px_#0DC5A0]" />
        OncoFlow
      </span>

      {showSearch && (
        <div className="flex-1 max-w-[360px] bg-surface border border-border2 h-[34px] flex items-center px-3 gap-2">
          <svg className="w-3.5 h-3.5 text-text3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            value={searchValue ?? ''}
            onChange={e => onSearchChange?.(e.target.value)}
            placeholder="Search by patient name or ID..."
            className="bg-transparent border-none outline-none text-text1 text-[13px] placeholder-text3 font-sans w-full"
          />
        </div>
      )}

      <div className="ml-auto flex items-center gap-2.5">
        {cta}
        {physician && (
          <div className="w-[30px] h-[30px] bg-surface2 border border-border2 flex items-center justify-center font-mono text-[10px] text-text2">
            {physician.initials}
          </div>
        )}
        <button
          onClick={handleLogout}
          className="w-[30px] h-[30px] bg-surface2 border border-border2 flex items-center justify-center text-text3 hover:text-danger transition-colors"
          title="Sign out"
        >
          <LogOut size={13} />
        </button>
      </div>
    </header>
  )
}
