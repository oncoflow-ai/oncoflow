export default function SkeletonRow() {
  return (
    <tr className="border-b border-border animate-pulse">
      {Array.from({ length: 7 }).map((_, i) => (
        <td key={i} className="px-3 py-4">
          <div className="h-3 bg-surface3 rounded-sm w-full" />
        </td>
      ))}
    </tr>
  )
}
