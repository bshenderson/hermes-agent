// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router';
import { SessionsRedirect } from '../components/SessionsRedirect';
import { expect, it, vi } from 'vitest';
import { ProfileProvider } from './ProfileProvider';
vi.mock('@/lib/api', () => ({ api: {
  getProfiles: () => new Promise(() => {}), getActiveProfile: () => new Promise(() => {}),
}, setManagementProfile: () => {} }));
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

it.each([
  '/hermes-dashboard/',
  '/hermes-dashboard',
  '/hermes-dashboard?profile=default',
  '/hermes-dashboard/?keep=yes',
  '/hermes-dashboard/unknown',
])('lands on sessions with synchronized scope from %s', async (entry) => {
  window.__HERMES_DASHBOARD_PROFILE__ = 'default';
  window.history.replaceState(null, '', entry);
  const container = document.createElement('div'); document.body.append(container);
  const root = createRoot(container);
  try {
    await act(async () => root.render(
      <BrowserRouter basename='/hermes-dashboard'><ProfileProvider><Routes>
        <Route path='/' element={<SessionsRedirect />} />
        <Route path='/sessions' element={<div>Session content</div>} />
        <Route path='*' element={<SessionsRedirect />} />
      </Routes></ProfileProvider></BrowserRouter>,
    ));
    expect(window.location.pathname).toBe('/hermes-dashboard/sessions');
    expect(container.textContent).toBe('Session content');
    expect(new URLSearchParams(window.location.search).get('profile')).toBe('default');
    if (entry.includes('keep=yes')) {
      expect(new URLSearchParams(window.location.search).get('keep')).toBe('yes');

    }
  } finally {
    await act(async () => root.unmount()); container.remove();
    delete window.__HERMES_DASHBOARD_PROFILE__;
  }
});
