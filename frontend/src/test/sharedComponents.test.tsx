import { render, screen } from '@testing-library/react'
import DeltaTag from '@/components/shared/DeltaTag'
import EmptyState from '@/components/shared/EmptyState'

describe('DeltaTag', () => {
  it('renders baseline when value is null', () => {
    render(<DeltaTag value={null} />)
    expect(screen.getByText(/Baseline/)).toBeInTheDocument()
  })
  it('renders positive delta', () => {
    render(<DeltaTag value={12.5} />)
    expect(screen.getByText(/12\.5/)).toBeInTheDocument()
  })
  it('renders negative delta', () => {
    render(<DeltaTag value={-8.3} />)
    expect(screen.getByText(/8\.3/)).toBeInTheDocument()
  })
})

describe('EmptyState', () => {
  it('renders title and description', () => {
    render(<EmptyState title="No data" description="Try again later" />)
    expect(screen.getByText('No data')).toBeInTheDocument()
    expect(screen.getByText('Try again later')).toBeInTheDocument()
  })
})
