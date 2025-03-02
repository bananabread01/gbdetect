import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import Router from '@/pages/Router'
import './globals.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Router />
  </StrictMode>,
)
