{{/*
通用命名与标签辅助模板。

命名策略：
- 默认用 "<release>-<chart>" 作为全名前缀，避免多 release 冲突；
- 若用户设置了 fullnameOverride 则完全采用用户值。
*/}}

{{- define "aetherserve.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "aetherserve.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "aetherserve.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* 标准标签：使用 app.kubernetes.io/* 惯例，便于生态工具发现 */}}
{{- define "aetherserve.labels" -}}
helm.sh/chart: {{ include "aetherserve.chart" . }}
{{ include "aetherserve.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/component: operator
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "aetherserve.selectorLabels" -}}
app.kubernetes.io/name: {{ include "aetherserve.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "aetherserve.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "aetherserve.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "aetherserve.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) -}}
{{- end -}}
