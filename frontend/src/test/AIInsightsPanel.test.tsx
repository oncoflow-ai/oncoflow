import { render, screen } from '@testing-library/react'
import AIInsightsPanel from '@/components/shared/AIInsightsPanel'
import { mockSummaries } from '@/data/mockData'

describe('AIInsightsPanel', () => {
  it('renders model name', () => {
    render(<AIInsightsPanel summary={mockSummaries['P-1029']} />)
    expect(screen.getByText(/MedGemma/)).toBeInTheDocument()
  })

  it('renders AI Clinical Narrative header', () => {
    render(<AIInsightsPanel summary={mockSummaries['P-1029']} />)
    expect(screen.getByText(/AI.*Clinical Narrative/i)).toBeInTheDocument()
  })

  it('renders multi-agent and RAG stream indicators', () => {
    render(<AIInsightsPanel summary={mockSummaries['P-1029']} />)
    expect(screen.getByText(/Image Stream/i)).toBeInTheDocument()
    expect(screen.getByText(/Prior Summary RAG/i)).toBeInTheDocument()
  })
})

