# Coding Conventions

**Analysis Date:** 2026-04-08

## Naming Patterns

**Files:**
- React components: PascalCase with `.tsx` extension - `StatBlock.tsx`, `PatientTable.tsx`, `AIInsightsPanel.tsx`
- Utilities: camelCase with `.ts` extension - `utils.ts`, `client.ts`
- Test files: Match source name with `.test.ts` or `.test.tsx` suffix - `utils.test.ts`, `AuthPage.test.tsx`
- Store files: camelCase with `Store` suffix - `authStore.ts`
- API modules: camelCase plural - `patients.ts`, `scans.ts`, `reports.ts`
- Config files: lowercase with extension - `tailwind.config.ts`, `vite.config.ts`, `tsconfig.json`

**Functions:**
- Components: PascalCase function declarations - `export default function StatBlock({ ... })`
- Utilities: camelCase function declarations - `export function formatDate()`, `export function cn()`
- Async functions: camelCase with async/await - `async function handleSignIn(e: React.FormEvent)`
- Event handlers: `handle` prefix - `handleSignIn`, `onSearchChange`

**Variables:**
- Local state: camelCase - `const [mode, setMode] = useState('signin')`
- Constants: camelCase or SCREAMING_SNAKE_CASE for arrays/objects - `const COLUMNS = [...]`
- Environment variables: `VITE_` prefix with SCREAMING_SNAKE_CASE - `VITE_API_URL`

**Types:**
- Interfaces: PascalCase - `interface Patient`, `interface AuthState`
- Type aliases: PascalCase - `type Mode = 'signin' | 'register'`
- Props interfaces: Component name + `Props` suffix - `interface StatBlockProps`
- Enum-like unions: PascalCase type name - `type PatientStatus = 'active' | 'review'`

## Code Style

**Formatting:**
- No automated formatter config detected (no `.prettierrc` or ESLint formatting rules)
- Indentation: 2 spaces (observed consistently)
- Quotes: Single quotes for strings (except JSX attributes)
- Semicolons: Present on all statements
- Trailing commas: Used in multi-line arrays and objects
- Line length: No strict limit enforced, typically < 100 characters

**Linting:**
- Tool: ESLint (via `package.json` script)
- Command: `npm run lint`
- Config: Not explicitly found (may be using defaults)
- Strict TypeScript enabled in `tsconfig.json`:
  - `"strict": true`
  - `"noUnusedLocals": true`
  - `"noUnusedParameters": true`
  - `"noFallthroughCasesInSwitch": true`

## Import Organization

**Order:**
1. External React imports - `import React from 'react'`, `import { useState } from 'react'`
2. External library imports - `import { useQuery } from '@tanstack/react-query'`
3. Router imports - `import { useNavigate } from 'react-router-dom'`
4. Internal store/state - `import { useAuthStore } from '@/store/authStore'`
5. Internal components - `import TopNav from '@/components/layout/TopNav'`
6. Internal utilities - `import { cn, formatDate } from '@/lib/utils'`
7. Internal API - `import { getPatients } from '@/api/patients'`
8. Internal types - `import type { Patient } from '@/types'`
9. Internal data - `import { mockPatients } from '@/data/mockData'`
10. Icons - `import { LogOut, Users } from 'lucide-react'`
11. CSS imports - `import './index.css'` (last)

**Path Aliases:**
- `@/*` maps to `src/*` (configured in `vite.config.ts` and `tsconfig.json`)
- Always use `@/` prefix for internal imports - `import { cn } from '@/lib/utils'`
- Relative imports only for sibling files - `import DeltaTag from './DeltaTag'`

**Type Imports:**
- Use `import type` for type-only imports - `import type { Patient, Scan } from '@/types'`

## Error Handling

**Patterns:**
- Throw `Error` objects with descriptive messages - `throw new Error('Patient ${id} not found')`
- Use try-catch in async event handlers:
```typescript
try {
  await login(physicianId, password)
  navigate('/dashboard')
} catch (err) {
  setError(err instanceof Error ? err.message : 'Sign in failed')
} finally {
  setLoading(false)
}
```
- Check `err instanceof Error` before accessing `.message`
- Display errors via state hooks - `const [error, setError] = useState('')`
- React Query handles API errors via `isError` flag

**API Errors:**
- Functions throw errors for not-found cases - `throw new Error('Scan ${scanId} not found')`
- Consumers catch or let React Query handle via `isError` state

## Logging

**Framework:** Built-in `console` (no logging library)

**Patterns:**
- No `console.log()` statements found in production code
- Appears to be avoided in favor of UI error states

## Comments

**When to Comment:**
- Inline explanations for non-obvious logic - `// Mock phase: accept any non-empty credentials`
- Type/purpose documentation above functions - `/** Format a date string (ISO) to "MMM DD, YYYY" */`
- Commented-out future features with context - `// JWT interceptor stub — no-op in mock phase.`

**JSDoc/TSDoc:**
- Used minimally for utility functions in `lib/utils.ts`:
```typescript
/** shadcn/ui cn() helper — merge Tailwind classes */
export function cn(...inputs: ClassValue[]) { ... }
```
- Not used on React components (rely on TypeScript types)

**Comment Style:**
- Single-line: `// Comment text`
- Multi-line: Use single-line style for short blocks
- JSDoc: `/** Comment text */` for exported utilities

## Function Design

**Size:**
- Components: 50-150 lines typical, max ~177 lines (`PatientDetailPage.tsx`)
- Utilities: 5-10 lines each in `lib/utils.ts`
- Event handlers: 10-20 lines inline in components
- API functions: 5-10 lines

**Parameters:**
- Components: Props destructured in signature - `function StatBlock({ label, value, delta, ... })`
- Use TypeScript interfaces for props - `interface StatBlockProps { label: string; ... }`
- Optional parameters with default values or `?` - `loading = false`, `delta?: number | null`
- Utilities: Positional parameters - `function formatDate(dateStr: string): string`

**Return Values:**
- Components: JSX.Element (implicit)
- Utilities: Explicit return types - `function formatVolume(mm3: number): string`
- Async functions: Promise types - `async function getScans(patientId: string): Promise<Scan[]>`
- Use early returns for null/empty states in React components

## Module Design

**Exports:**
- Components: Default export - `export default function StatBlock(...)`
- Utilities: Named exports - `export function cn(...)`, `export function formatDate(...)`
- Types: Named exports from central file - `export type PatientStatus = ...` in `types/index.ts`
- Constants: Named exports - `export const apiClient = ...`
- Store hooks: Named exports - `export const useAuthStore = create<AuthState>()(...)`

**Barrel Files:**
- Central types file: `src/types/index.ts` exports all type definitions
- Mock data: `src/data/mockData.ts` exports all fixtures
- No barrel files for components (direct imports)

**File Organization:**
- One component per file
- Co-locate helpers only if component-specific
- Shared utilities in `lib/` directory
- Store files self-contained with state and actions

## React Patterns

**State Management:**
- Local state: `useState` hook - `const [search, setSearch] = useState('')`
- Global state: Zustand stores - `export const useAuthStore = create<AuthState>()(...)` in `store/authStore.ts`
- Server state: React Query - `const { data, isLoading, isError } = useQuery({ ... })`
- Form state: Controlled components with local `useState`

**Component Structure:**
```typescript
import { useState } from 'react'
import type { PropsType } from '@/types'
import { helperFunction } from '@/lib/utils'

interface ComponentProps {
  prop: string
  optional?: number
}

export default function Component({ prop, optional = 0 }: ComponentProps) {
  const [state, setState] = useState(initialValue)
  
  // Event handlers
  async function handleEvent() { ... }
  
  // Early returns for conditional rendering
  if (!data) return <EmptyState />
  
  // Main render
  return (
    <div className={cn('base-classes', conditionalClass)}>
      {/* JSX content */}
    </div>
  )
}
```

**Conditional Rendering:**
- Ternary for simple cases - `{loading ? <Spinner /> : <Content />}`
- Logical AND for optional rendering - `{error && <ErrorBanner />}`
- Early returns for complex conditions
- Nullish coalescing for defaults - `data ?? []`

**Styling:**
- Tailwind CSS classes - `className="bg-surface px-4 py-2"`
- `cn()` helper for conditional classes - `cn('base-class', condition && 'conditional-class')`
- No CSS modules or styled-components
- Custom colors in `tailwind.config.ts` - `bg-surface`, `text-text1`, `border-border`

---

*Convention analysis: 2026-04-08*
