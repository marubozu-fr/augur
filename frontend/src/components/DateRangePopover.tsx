import { useState } from 'react'
import { Popover } from '@mantine/core'
import { DatePicker } from '@mantine/dates'
import dayjs from 'dayjs'
import styles from './DateRangePopover.module.css'

type DraftRange = [Date | null, Date | null]

interface DateRangePopoverProps {
  /** True when the Custom preset is the active period (highlights the trigger). */
  active: boolean
  /** Applied custom range as inclusive ISO dates, or [null, null] when unset. */
  value: [string | null, string | null]
  /** Earliest selectable date (data_range[0] of the current timeframe). */
  minDate?: Date
  /** Latest selectable date (data_range[1] of the current timeframe). */
  maxDate?: Date
  /** Fired with inclusive ISO start/end when the user applies a complete range. */
  onApply: (start: string, end: string) => void
}

function toDate(iso: string | null): Date | null {
  if (!iso) return null
  const d = dayjs(iso)
  return d.isValid() ? d.toDate() : null
}

function fmt(d: Date | null): string {
  return d ? dayjs(d).format('YYYY-MM-DD') : '—'
}

/**
 * "Custom" trigger + popover holding a two-month range calendar. The draft
 * selection is local, so Cancel discards it; Apply lifts the completed range up
 * and closes. The calendar is clamped to the timeframe's available data range.
 */
export function DateRangePopover({
  active,
  value,
  minDate,
  maxDate,
  onApply,
}: DateRangePopoverProps) {
  const [opened, setOpened] = useState(false)
  const [draft, setDraft] = useState<DraftRange>([null, null])

  function open() {
    // Seed the draft from the applied range each time the popover opens.
    setDraft([toDate(value[0]), toDate(value[1])])
    setOpened(true)
  }

  function cancel() {
    setOpened(false)
  }

  function apply() {
    const [start, end] = draft
    if (!start || !end) return
    onApply(dayjs(start).format('YYYY-MM-DD'), dayjs(end).format('YYYY-MM-DD'))
    setOpened(false)
  }

  const complete = draft[0] !== null && draft[1] !== null

  return (
    <Popover
      opened={opened}
      onChange={setOpened}
      position="bottom-end"
      withArrow
      shadow="md"
      trapFocus
    >
      <Popover.Target>
        <button
          type="button"
          className={`${styles.trigger} ${active ? styles.triggerActive : ''}`}
          onClick={() => (opened ? cancel() : open())}
        >
          <span className={styles.triggerIcon} aria-hidden="true">
            &#9638;
          </span>
          <span>Custom</span>
          <span className={styles.triggerCaret} aria-hidden="true">
            &#9662;
          </span>
        </button>
      </Popover.Target>

      <Popover.Dropdown>
        <div className={styles.panel}>
          <DatePicker
            type="range"
            numberOfColumns={2}
            value={draft}
            onChange={setDraft}
            minDate={minDate}
            maxDate={maxDate}
            allowSingleDateInRange
          />
          <div className={styles.foot}>
            <span className={styles.rangeDisplay}>
              <span className={styles.rangeNum}>{fmt(draft[0])}</span>
              {' → '}
              <span className={styles.rangeNum}>{fmt(draft[1])}</span>
            </span>
            <div className={styles.actions}>
              <button
                type="button"
                className={styles.btnGhost}
                onClick={cancel}
              >
                Cancel
              </button>
              <button
                type="button"
                className={styles.btnPrimary}
                onClick={apply}
                disabled={!complete}
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      </Popover.Dropdown>
    </Popover>
  )
}
