import { useParams } from 'react-router-dom'
import styles from './PlaceholderPage.module.css'

export function StatDetailPage() {
  const { family } = useParams<{ family: string }>()

  return (
    <div className={styles.placeholder}>
      <p className={styles.placeholderTitle}>{family}</p>
      <p className={styles.placeholderText}>
        Stat detail visualizations will be implemented in issue #108.
      </p>
    </div>
  )
}
