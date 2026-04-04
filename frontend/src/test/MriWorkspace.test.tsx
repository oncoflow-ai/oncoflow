import { render, screen } from '@testing-library/react'
import MriWorkspace from '@/components/shared/MriWorkspace'
import { mockScans } from '@/data/mockData'

const scan = mockScans['P-1029'][0]

describe('MriWorkspace', () => {
  it('renders study label', () => {
    render(<MriWorkspace scan={scan} />)
    expect(screen.getByText('MRI Study #1')).toBeInTheDocument()
  })

  it('shows slice count in metadata', () => {
    render(<MriWorkspace scan={scan} />)
    expect(screen.getByText(/128 SL/)).toBeInTheDocument()
  })

  it('brush tool is active by default', () => {
    render(<MriWorkspace scan={scan} />)
    // the Brush annotation button should exist
    expect(screen.getByTitle('Brush annotation')).toBeInTheDocument()
  })
})
