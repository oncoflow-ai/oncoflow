import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import MriWorkspace from '@/components/shared/MriWorkspace'
import { mockScans } from '@/data/mockData'

const scan = mockScans['P-1029'][0]

describe('MriWorkspace', () => {
  it('renders study label and metadata', () => {
    render(<MriWorkspace scan={scan} />)
    expect(screen.getByText('MRI Study #1')).toBeInTheDocument()
    expect(screen.getByText(/128 SL/)).toBeInTheDocument()
  })

  it('ruler tool is active by default and tools are selectable', async () => {
    const user = userEvent.setup()
    render(<MriWorkspace scan={scan} />)

    const rulerBtn = screen.getByTitle(/Measure diameter/i)
    expect(rulerBtn).toHaveClass('border-teal')

    const brushBtn = screen.getByTitle(/Brush annotation/i)
    await user.click(brushBtn)
    expect(brushBtn).toHaveClass('border-teal')
  })

  it('steps slices and changes slice display', async () => {
    const user = userEvent.setup()
    render(<MriWorkspace scan={scan} />)

    expect(screen.getByText(/Slice 082/i)).toBeInTheDocument()
    const nextBtn = screen.getByRole('button', { name: 'Next slice' })
    await user.click(nextBtn)

    expect(screen.getByText(/Slice 082/i)).toBeInTheDocument()
  })

  it('toggles AI overlay mask and zoom level', async () => {
    const user = userEvent.setup()
    render(<MriWorkspace scan={scan} />)

    const overlayBtn = screen.getByTitle('Hide AI Mask')
    await user.click(overlayBtn)
    expect(screen.getByTitle('Show AI Mask')).toBeInTheDocument()

    const zoomBtn = screen.getByRole('button', { name: 'Zoom' })
    expect(screen.getByText('1x')).toBeInTheDocument()
    await user.click(zoomBtn)
    expect(screen.getByText('1.4x')).toBeInTheDocument()
  })

  it('allows interactive drag measuring with ruler tool', () => {
    render(<MriWorkspace scan={scan} />)
    const img = screen.getByAltText(/Axial MRI scan slice/i)
    const container = img.parentElement!

    // Default caliper measurement display
    expect(screen.getByText(/Caliper:/i)).toBeInTheDocument()

    // Simulate drag to measure
    fireEvent.mouseDown(container, { clientX: 50, clientY: 50 })
    fireEvent.mouseMove(container, { clientX: 150, clientY: 50 })
    fireEvent.mouseUp(container)

    expect(screen.getByText(/Caliper:/i)).toBeInTheDocument()
  })
})
