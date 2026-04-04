import { render, screen } from '@testing-library/react'
import StatBlock from '@/components/shared/StatBlock'

describe('StatBlock', () => {
  it('renders label and value', () => {
    render(<StatBlock label="Total Scans" value="3" />)
    expect(screen.getByText('Total Scans')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('renders delta tag when delta provided', () => {
    render(<StatBlock label="Volume" value="12,480 mm³" delta={-12.5} deltaUnit="%" />)
    expect(screen.getByText(/12\.5/)).toBeInTheDocument()
  })

  it('renders custom badge when badge provided', () => {
    render(<StatBlock label="Status" value="" badge={<span>Custom Badge</span>} />)
    expect(screen.getByText('Custom Badge')).toBeInTheDocument()
  })
})
