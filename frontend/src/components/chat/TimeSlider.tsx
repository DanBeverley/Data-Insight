import React from 'react';
import { Slider } from "@/components/ui/slider";
import { Clock } from "lucide-react";

interface TimeSliderProps {
    value: number;
    onChange: (value: number) => void;
    min?: number;
    max?: number;
}

export const TimeSlider: React.FC<TimeSliderProps> = ({
    value,
    onChange,
    min = 5,
    max = 60
}) => {
    const formatTime = (mins: number) => {
        if (mins < 60) return `${mins}m`;
        return `${Math.floor(mins / 60)}h ${mins % 60}m`;
    };

    return (
        <div className="time-slider-container" style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '8px 12px',
            background: 'rgba(0, 255, 255, 0.05)',
            borderRadius: '8px',
            border: '1px solid rgba(0, 255, 255, 0.2)'
        }}>
            <Clock size={16} style={{ color: 'rgba(0, 255, 255, 0.7)' }} />
            <div style={{ flex: 1 }}>
                <Slider
                    value={[value]}
                    onValueChange={(vals) => onChange(vals[0])}
                    min={min}
                    max={max}
                    step={5}
                    className="research-time-slider"
                />
            </div>
            <span style={{
                minWidth: '50px',
                textAlign: 'right',
                fontSize: '13px',
                fontWeight: 500,
                color: 'rgba(0, 255, 255, 0.9)'
            }}>
                {formatTime(value)}
            </span>
        </div>
    );
};

export default TimeSlider;
