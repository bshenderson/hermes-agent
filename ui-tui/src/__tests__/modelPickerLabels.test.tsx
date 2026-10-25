import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { expect, it, vi } from 'vitest'

import { ModelPicker } from '../components/modelPicker.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_THEME } from '../theme.js'

it('requests configured providers, renders human labels, searches IDs and selects the wire ID', async () => {
  const request = vi.fn(async () => ({ providers: [{ slug: 'main', name: 'Main', models: ['wire-a'],
    model_labels: { 'wire-a': 'Human Base Model' }, capabilities: { 'wire-a': { reasoning: false } } }] }))
  const onSelect = vi.fn()
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 30, isTTY: false })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: () => {}, ref: () => {}, unref: () => {} })
  let output = ''
  stdout.on('data', chunk => { output += stripAnsi(chunk.toString()) })
  const view = renderSync(<ModelPicker gw={{ request } as unknown as GatewayClient} onCancel={() => {}}
    onSelect={onSelect} t={DEFAULT_THEME} />, {
    stdout: stdout as unknown as NodeJS.WriteStream, stdin: stdin as unknown as NodeJS.ReadStream,
    stderr: new PassThrough() as unknown as NodeJS.WriteStream, patchConsole: false
  })
  try {
    await vi.waitFor(() => expect(output).toContain('Main'))
    expect(request).toHaveBeenCalledWith('model.options', { include_unconfigured: false })
    stdin.write('\r')
    await vi.waitFor(() => expect(output).toContain('Human Base Model'))
    for (const ch of 'wire-a') {
      stdin.write(ch)
      await new Promise(resolve => setTimeout(resolve, 20))
    }
    await vi.waitFor(() => expect(output).toContain('wire-a'))
    stdin.write('\r')
    await vi.waitFor(() => expect(onSelect).toHaveBeenCalledWith('wire-a --provider main --tui-session'))
  } finally {
    view.unmount()
    view.cleanup()
  }
})
