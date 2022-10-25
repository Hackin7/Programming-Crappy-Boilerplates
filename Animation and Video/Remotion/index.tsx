/*

Usage
* npm init video - Use Base Template
* Replace index.tsx with this file
* npm start

*/

//https://www.remotion.dev/docs/reusability 
import {registerRoot} from 'remotion';
import {Composition} from 'remotion';

//// Component /////////////////////////////////////////////////////////
import { 
  AbsoluteFill, useVideoConfig, useCurrentFrame,
  Sequence,
  interpolate, spring,
  Video 
} from "remotion";

const Title: React.FC<{title: string}> = ({title}) => {
    const frame = useCurrentFrame()
    const opacity = interpolate(frame, [0, 20], [0, 1], {extrapolateRight: 'clamp'})
 
    return (
      <div style={{opacity}}>{title}</div>
    )
}
 
export const MyComposition = () => {
  const { fps, durationInFrames, width, height } = useVideoConfig();
  const frame = useCurrentFrame();
	return <AbsoluteFill
      style={{
        justifyContent: "center",
        alignItems: "center",
        fontSize: 30,
        backgroundColor: "#0F0E18",
        color: "white",
      }}
    >
      <div>This {width}x{height}px video is {durationInFrames / fps} seconds long.</div>
      
      <div style={{opacity: frame / durationInFrames}}>Animate Opacity</div>
      <div style={{
        opacity: interpolate(frame, [0, durationInFrames], [0, 1], {
          extrapolateRight: "clamp",
        })
      }}>Animate Opacity - Linear interpolate</div>
      <div style={{
          transform: `scale(${ spring({fps,frame,}) })`
        }}>Animate Scale - Spring</div>
        
      <div>
        <Sequence from={0} durationInFrames={40}>
          <Title title="Hello" />
        </Sequence>
        <Sequence from={40}>
          <Title title="World" />
          <Video src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" />
        </Sequence>
      </div>
    </AbsoluteFill>;
};

//// Video component //////////////////////////////////////////////////
export const RemotionVideo: React.FC = () => {
	return (
		<>
			<Composition
				id="MyComp"
				component={MyComposition}
				durationInFrames={60}
				fps={30}
				width={1280}
				height={720}
			/>
		</>
	);
};


//registerRoot(RemotionVideo);
