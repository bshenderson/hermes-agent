import { Navigate, useLocation } from "react-router";

/** Preserve scope while redirecting: profile URL synchronization can navigate
 * in the same commit, so a changed query must also retrigger this redirect. */
export function SessionsRedirect() {
  const { search, hash } = useLocation();
  return <Navigate to={{ pathname: "/sessions", search, hash }} replace />;
}
