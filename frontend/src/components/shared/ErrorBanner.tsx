import { AlertCircle } from 'lucide-react'

interface ErrorBannerProps {
  message: string
  onRetry?: () => void
}

export default function ErrorBanner({ message, onRetry }: ErrorBannerProps) {
  return (
    <div className="flex items-center gap-3 bg-danger/10 border border-danger/30 px-4 py-3 text-danger text-[13px] font-sans">
      <AlertCircle size={15} />
      <span>{message}</span>
      {onRetry && (
        <button onClick={onRetry} className="ml-auto text-[11px] font-mono underline underline-offset-2 hover:no-underline">
          Retry
        </button>
      )}
    </div>
  )
}
