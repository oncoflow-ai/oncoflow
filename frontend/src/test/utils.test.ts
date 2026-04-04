import { cn, formatDate, formatVolume, formatDelta } from '@/lib/utils'

describe('cn()', () => {
  it('merges class names', () => {
    expect(cn('a', 'b')).toBe('a b')
  })
  it('deduplicates conflicting Tailwind classes', () => {
    expect(cn('p-2', 'p-4')).toBe('p-4')
  })
})

describe('formatDate()', () => {
  it('formats ISO date string', () => {
    expect(formatDate('2026-03-15')).toMatch(/Mar/)
  })
})

describe('formatVolume()', () => {
  it('formats with thousands separator', () => {
    expect(formatVolume(12480)).toBe('12,480')
  })
})

describe('formatDelta()', () => {
  it('prefixes positive with +', () => {
    expect(formatDelta(12.3)).toBe('+12.3%')
  })
  it('negative has no extra sign', () => {
    expect(formatDelta(-4.5)).toBe('-4.5%')
  })
})
