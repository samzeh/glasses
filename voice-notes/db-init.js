import { readFile } from 'node:fs/promises';
import { pool } from './db.js';

try {
  const schema = await readFile(new URL('./schema.sql', import.meta.url), 'utf8');
  await pool.query(schema);
  console.log('Voice Notes table is ready in TigerData.');
} catch (error) {
  console.error('Could not initialize Voice Notes:', error.message);
  process.exitCode = 1;
} finally {
  await pool.end();
}
