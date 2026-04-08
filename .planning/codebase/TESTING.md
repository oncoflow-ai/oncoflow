# Testing Patterns

**Analysis Date:** 2026-04-08

## Test Framework

**Runner:**
- Vitest 1.4.0
- Config: `vite.config.ts` (test block)

**Assertion Library:**
- Vitest (built-in expect)
- `@testing-library/jest-dom` for DOM matchers

**Run Commands:**
```bash
npm run test              # Run all tests once
npm run test -- --watch   # Watch mode (via Vitest)
# Coverage command not configured in package.json
```

**Configuration:**
```typescript
// vite.config.ts
test: {
  globals: true,
  environment: 'jsdom',
  setupFiles: ['./src/test/setup.ts'],
}
```

## Test File Organization

**Location:**
- All tests in `src/test/` directory (separate from source)
- Not co-located with source files

**Naming:**
- Pattern: `{ComponentOrModuleName}.test.ts` or `.test.tsx`
- Examples:
  - `utils.test.ts` for `lib/utils.ts`
  - `AIInsightsPanel.test.tsx` for `components/shared/AIInsightsPanel.tsx`
  - `authStore.test.ts` for `store/authStore.ts`
  - `api.test.ts` for all API modules

**Structure:**
```
src/
├── test/
│   ├── setup.ts                    # Test setup/config
│   ├── sanity.test.ts              # Basic sanity check
│   ├── utils.test.ts               # Utility tests
│   ├── types.test.ts               # Type validation tests
│   ├── authStore.test.ts           # Store tests
│   ├── api.test.ts                 # API function tests
│   ├── mockData.test.ts            # Mock data validation
│   ├── AIInsightsPanel.test.tsx    # Component tests
│   ├── StatBlock.test.tsx
│   ├── PatientTable.test.tsx
│   ├── MriWorkspace.test.tsx
│   ├── ImagingHistory.test.ts
│   ├── VolumeChart.test.tsx
│   ├── sharedComponents.test.tsx   # Multiple small components
│   ├── AuthPage.test.tsx           # Page tests
│   ├── DashboardPage.test.tsx
│   └── router.test.tsx             # Router/navigation tests
```

**Setup File:**
```typescript
// src/test/setup.ts
import '@testing-library/jest-dom'
```

## Test Structure

**Suite Organization:**
```typescript
import { render, screen } from '@testing-library/react'
import Component from '@/components/path/Component'

describe('ComponentName', () => {
  it('describes expected behavior', () => {
    render(<Component prop="value" />)
    expect(screen.getByText('Expected')).toBeInTheDocument()
  })
  
  it('describes another behavior', async () => {
    // Test code
  })
})
```

**Patterns:**
- Use `describe()` for component/module name
- Use `it()` for individual test cases (not `test()`)
- Test names are lowercase sentence fragments - `'renders model name'`, `'shows error on empty credentials'`
- Group related tests under same `describe()` block
- One `describe()` block per component/module

**Multiple Components in One File:**
```typescript
// sharedComponents.test.tsx
describe('DeltaTag', () => {
  it('renders baseline when value is null', () => { ... })
  it('renders positive delta', () => { ... })
})

describe('EmptyState', () => {
  it('renders title and description', () => { ... })
})
```

**Setup/Teardown:**
```typescript
beforeEach(() => {
  useAuthStore.setState({ physician: null })
})
```
- Use `beforeEach()` for test isolation
- Reset state between tests (especially Zustand stores)

## Mocking

**Framework:** Vitest built-in mocking

**Patterns:**

**No external mocking detected** - Tests use:
1. **Real mock data** from `@/data/mockData`:
```typescript
import { mockPatients, mockScans, mockSummaries } from '@/data/mockData'

it('returns 8 patients', async () => {
  const patients = await getPatients()
  expect(patients).toHaveLength(8)
})
```

2. **Real implementations** - API functions return mock data via delay:
```typescript
// api/scans.ts
export async function getScans(patientId: string): Promise<Scan[]> {
  await delay(350)
  return mockScans[patientId] ?? []
}
```

3. **State reset** rather than mocking stores:
```typescript
beforeEach(() => {
  useAuthStore.setState({ physician: null })
})
```

4. **Test-specific components** for isolation:
```typescript
// router.test.tsx
function Protected() {
  const physician = useAuthStore(s => s.physician)
  if (physician === null) return <div>Redirected to auth</div>
  return <div>Protected content</div>
}
```

**What to Mock:**
- Currently: Nothing mocked via vi.fn() or vi.mock()
- Pattern: Use real implementations with mock data

**What NOT to Mock:**
- API functions (use real ones with mock data)
- Zustand stores (use real store with state reset)
- Utility functions (use real implementations)
- React Router (wrap in `<MemoryRouter>` instead)

## Fixtures and Factories

**Test Data:**
```typescript
// data/mockData.ts exports fixtures
export const mockPatients: Patient[] = [ ... ]
export const mockScans: Record<string, Scan[]> = { ... }
export const mockSummaries: Record<string, Summary> = { ... }

// Used in tests
import { mockPatients, mockScans } from '@/data/mockData'

it('renders all 8 patient rows', () => {
  render(<PatientTable patients={mockPatients} scansMap={mockScans} />)
  expect(screen.getByText('Sarah Jenkins')).toBeInTheDocument()
})
```

**Inline Test Data:**
```typescript
// For simple cases, define inline
it('Patient shape is correct', () => {
  const p: Patient = {
    id: 'P-1029',
    name: 'Test',
    dob: '1994-07-22',
    diagnosis: 'Osteosarcoma',
    diagnosisLocation: 'Distal Left Femur',
    assignedPhysicianId: 'DR-001',
    status: 'active',
    scanCount: 3,
    lastScanDate: '2026-03-08',
  }
  expect(p.id).toBe('P-1029')
})
```

**Helper Functions:**
```typescript
// Reusable render helpers
function renderAuthPage() {
  return render(
    <MemoryRouter>
      <AuthPage />
    </MemoryRouter>
  )
}

function renderTable(loading = false) {
  render(
    <MemoryRouter>
      <PatientTable patients={mockPatients} scansMap={mockScans} loading={loading} />
    </MemoryRouter>
  )
}
```

**Location:**
- Shared fixtures: `src/data/mockData.ts`
- Test-specific helpers: Defined at top of test file

## Coverage

**Requirements:** None enforced (no coverage script in `package.json`)

**View Coverage:**
```bash
# No built-in command
# Manual: npx vitest run --coverage
```

**Current Status:**
- 13 test files for 45 total source files
- Test coverage: ~29% by file count
- Focus areas tested:
  - Core utilities (`utils.test.ts`)
  - Auth flow (`authStore.test.ts`, `AuthPage.test.tsx`)
  - API functions (`api.test.ts`)
  - Key components (`StatBlock`, `AIInsightsPanel`, `PatientTable`, `MriWorkspace`)
  - Type definitions (`types.test.ts`)
  - Router protection (`router.test.tsx`)

**Gaps:**
- No tests for: `TopNav`, `PatientRow`, `ScanRow`, `DeltaTag` (tested in `sharedComponents.test.tsx`), `ErrorBanner`, `SkeletonRow`, `EmptyState` (partially tested), `VolumeChart` (tested), `ImagingHistory` (tested), `PatientDetailPage`

## Test Types

**Unit Tests:**
- Utility functions tested in isolation:
```typescript
describe('formatDate()', () => {
  it('formats ISO date string', () => {
    expect(formatDate('2026-03-15')).toMatch(/Mar/)
  })
})
```
- Store actions tested with hooks:
```typescript
it('login sets physician', async () => {
  const { result } = renderHook(() => useAuthStore())
  await act(async () => {
    await result.current.login('DR-001', 'password')
  })
  expect(result.current.physician?.name).toBe('Dr. D. Cohen')
})
```

**Integration Tests:**
- API functions tested with mock data:
```typescript
describe('getPatients()', () => {
  it('returns 8 patients', async () => {
    const patients = await getPatients()
    expect(patients).toHaveLength(8)
  })
})
```
- Components tested with dependencies:
```typescript
it('renders all 8 patient rows', () => {
  renderTable()
  expect(screen.getByText('Sarah Jenkins')).toBeInTheDocument()
  expect(screen.getByText('David Levi')).toBeInTheDocument()
})
```

**E2E Tests:**
- Not used (no Playwright, Cypress, or E2E framework)
- Router tests verify navigation logic but don't test full flows

## Common Patterns

**Async Testing:**
```typescript
it('login sets physician', async () => {
  const { result } = renderHook(() => useAuthStore())
  await act(async () => {
    await result.current.login('DR-001', 'password')
  })
  expect(result.current.physician?.name).toBe('Dr. D. Cohen')
})
```
- Use `async/await` in test function
- Wrap state updates in `act()` from `@testing-library/react`
- Use `await screen.findByText()` for async appearance

**Error Testing:**
```typescript
it('login throws with empty credentials', async () => {
  const { result } = renderHook(() => useAuthStore())
  await expect(
    act(async () => result.current.login('', ''))
  ).rejects.toThrow('required')
})
```
- Use `await expect(...).rejects.toThrow()` for async errors
- Check error message substring with `.toThrow('substring')`

**Component Testing:**
```typescript
it('renders model name', () => {
  render(<AIInsightsPanel summary={mockSummaries['P-1029']} />)
  expect(screen.getByText(/MedGemma/)).toBeInTheDocument()
})
```
- Use `render()` from `@testing-library/react`
- Query with `screen.getByText()`, `screen.getByPlaceholderText()`, etc.
- Use regex for partial matches - `/MedGemma/`
- Assertion: `.toBeInTheDocument()` from `@testing-library/jest-dom`

**Negative Assertions:**
```typescript
it('shows skeleton rows when loading', () => {
  renderTable(true)
  expect(screen.queryByText('Sarah Jenkins')).not.toBeInTheDocument()
})
```
- Use `queryByText()` (not `getByText()`) for elements that shouldn't exist
- Assert with `.not.toBeInTheDocument()`

**User Interaction:**
```typescript
import userEvent from '@testing-library/user-event'

it('toggles to register mode', async () => {
  renderAuthPage()
  await userEvent.click(screen.getByText(/Request access from admin/))
  expect(screen.getByText(/Request Access/i)).toBeInTheDocument()
})
```
- Import `userEvent` from `@testing-library/user-event`
- Use `await userEvent.click()` for clicks
- Query result with `screen.findByText()` or `screen.getByText()`

**Router Testing:**
```typescript
render(
  <MemoryRouter initialEntries={['/dashboard']}>
    <Routes>
      <Route path="/dashboard" element={<Protected />} />
    </Routes>
  </MemoryRouter>
)
```
- Wrap components in `<MemoryRouter>` from `react-router-dom`
- Use `initialEntries` to set route
- Define `<Routes>` and `<Route>` for routing logic

**Hook Testing:**
```typescript
import { renderHook, act } from '@testing-library/react'

const { result } = renderHook(() => useAuthStore())
expect(result.current.physician).toBeNull()
```
- Use `renderHook()` for custom hooks and Zustand stores
- Access hook value via `result.current`
- Wrap mutations in `act()`

**Type Testing:**
```typescript
// types.test.ts
it('Patient shape is correct', () => {
  const p: Patient = { /* all fields */ }
  expect(p.id).toBe('P-1029')
})
```
- Create instances to verify TypeScript types compile
- Minimal assertions (type checking is primary goal)

**Data Validation:**
```typescript
// mockData.test.ts
it('each patient has required fields', () => {
  mockPatients.forEach(p => {
    expect(p.id).toBeTruthy()
    expect(p.name).toBeTruthy()
    expect(p.diagnosis).toBeTruthy()
  })
})
```
- Validate fixture data structure
- Ensure mock data is consistent

---

*Testing analysis: 2026-04-08*
