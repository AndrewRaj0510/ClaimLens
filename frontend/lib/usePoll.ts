"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Load something, and keep reloading every `intervalMs` while `keepPolling`
 * says so (e.g. while a claim is processing). Stops on unmount and pauses
 * while the tab is hidden.
 */
export function usePoll<T>(
  load: () => Promise<T>,
  keepPolling: (data: T) => boolean,
  intervalMs = 2000,
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const loadRef = useRef(load);
  const keepRef = useRef(keepPolling);
  useEffect(() => {
    loadRef.current = load;
    keepRef.current = keepPolling;
  });

  const refresh = useCallback(async () => {
    try {
      const next = await loadRef.current();
      setData(next);
      setError(null);
      return next;
    } catch (e) {
      setError(e as Error);
      return null;
    }
  }, []);

  // Bumped by restart(), e.g. after a decision, to resume polling.
  const [generation, setGeneration] = useState(0);
  const restart = useCallback(() => setGeneration((g) => g + 1), []);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    let stopped = false;
    const tick = async () => {
      if (document.hidden) {
        timer = setTimeout(tick, intervalMs);
        return;
      }
      const next = await refresh();
      if (stopped) return;
      // Keep trying after an error too (a blip in the network).
      if (next === null || keepRef.current(next)) timer = setTimeout(tick, intervalMs);
    };
    tick();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [refresh, intervalMs, generation]);

  return { data, error, refresh, restart };
}
