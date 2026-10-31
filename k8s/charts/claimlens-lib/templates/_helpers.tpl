{{/*
Shared by every ClaimLens chart. A service chart's templates/app.yaml is one
line, {{ include "claimlens.app" . }}, and its values.yaml says what's
different about it. See k8s/README.md for the values each template reads.
*/}}

{{- define "claimlens.name" -}}
{{- .Values.nameOverride | default .Chart.Name -}}
{{- end }}

{{- define "claimlens.selectorLabels" -}}
app.kubernetes.io/name: {{ include "claimlens.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "claimlens.labels" -}}
{{ include "claimlens.selectorLabels" . }}
app.kubernetes.io/part-of: claimlens
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/* Values shared by every chart, set once in the umbrella chart. */}}
{{- define "claimlens.global" -}}
{{- toYaml (.Values.global | default dict) -}}
{{- end }}

{{/*
Our images: <global.image.registry>/<name>:<global.image.tag>, e.g.
claimlens/gateway:dev on kind or ghcr.io/<owner>/claimlens/gateway:<sha>.
Third-party images (postgres, redis, ...) set image.repository and their own
pinned image.tag, and never follow the global tag.
*/}}
{{- define "claimlens.image" -}}
{{- $g := fromYaml (include "claimlens.global" .) -}}
{{- $gi := $g.image | default dict -}}
{{- if .Values.image.repository -}}
{{ .Values.image.repository }}:{{ required "image.tag is required with image.repository" .Values.image.tag }}
{{- else -}}
{{ $gi.registry | default "claimlens" }}/{{ .Values.image.name | default (include "claimlens.name" .) }}:{{ .Values.image.tag | default $gi.tag | default "dev" }}
{{- end -}}
{{- end }}

{{- define "claimlens.pullPolicy" -}}
{{- $g := fromYaml (include "claimlens.global" .) -}}
{{- .Values.image.pullPolicy | default (dig "image" "pullPolicy" "IfNotPresent" $g) -}}
{{- end }}

{{- define "claimlens.secretName" -}}
{{- $g := fromYaml (include "claimlens.global" .) -}}
{{- $g.secretName | default "claimlens-env" -}}
{{- end }}

{{/*
Environment: secrets first, then plain values, so plain values can use
$(VAR) to build URLs from a secret, e.g.
  secretEnv:  {DB_PASSWORD: DB_PASSWORD_GATEWAY}
  env:        {DATABASE_URL: "postgresql://claimlens_gateway:$(DB_PASSWORD)@postgres:5432/claimlens"}
Each pod gets only the secret keys it names, not the whole .env.
A secretEnv entry is a key name, or {key, optional} for keys that may be unset.
*/}}
{{- define "claimlens.env" -}}
{{- $secret := include "claimlens.secretName" . -}}
{{- range $name, $ref := .Values.secretEnv }}
- name: {{ $name }}
  valueFrom:
    secretKeyRef:
      name: {{ $secret }}
      {{- if kindIs "string" $ref }}
      key: {{ $ref }}
      {{- else }}
      key: {{ $ref.key }}
      optional: {{ $ref.optional | default false }}
      {{- end }}
{{- end }}
{{- range $name, $value := .Values.env }}
- name: {{ $name }}
  value: {{ $value | toString | quote }}
{{- end }}
{{- with .Values.cpuThreads }}
{{- /* ONNX Runtime and OpenCV size their thread pools from the node's cores, not the CPU limit. */}}
- name: ORT_THREADS
  value: {{ . | quote }}
{{- end }}
{{- end }}

{{/* Pod and container hardening: non-root, no privilege escalation, no capabilities. */}}
{{- define "claimlens.podSecurityContext" -}}
{{- if .Values.podSecurityContext -}}
{{ toYaml .Values.podSecurityContext }}
{{- else -}}
runAsNonRoot: true
# The images create their user by name (useradd appuser / node); Kubernetes
# can only verify "non-root" for a numeric UID, so it's stated here.
runAsUser: 1000
runAsGroup: 1000
fsGroup: 1000
seccompProfile:
  type: RuntimeDefault
{{- end -}}
{{- end }}

{{- define "claimlens.containerSecurityContext" -}}
{{- if .Values.containerSecurityContext -}}
{{ toYaml .Values.containerSecurityContext }}
{{- else -}}
allowPrivilegeEscalation: false
capabilities:
  drop: [ALL]
{{- end -}}
{{- end }}
