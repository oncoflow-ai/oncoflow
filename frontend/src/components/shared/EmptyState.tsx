import { cn } from '@/lib/utils'

interface EmptyStateProps {
  icon?: React.ReactNode
  title: string
  description?: string
  action?: React.ReactNode
  className?: string
}

export default function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center text-center py-16 gap-3', className)}>
      {icon && <div className="text-text3 mb-1">{icon}</div>}
      <p className="text-[14px] font-sans font-semibold text-text2">{title}</p>
      {description && <p className="text-[12px] text-text3 font-sans max-w-xs">{description}</p>}
      {action}
    </div>
  )
}
